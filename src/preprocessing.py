from pathlib import Path
import pandas as pd
from pandas import DataFrame
from typing import List
from transformers import AutoTokenizer

from filtering import remove_urls


def read_raw_data_to_df(file_name: str) -> DataFrame:
    path_to_file = Path(__file__).parents[1] / 'datalake' / file_name
    df = pd.read_csv(path_to_file, sep = '\t')
    return df


def prepare_df(df: DataFrame, col_with_text: str, n_rows = None) -> DataFrame:
    """
    Function takes in a DataFrame, name of coulumn containing text and optionally number of rows to sample,
    dropps all rows with NaN,
    samples random n number of rows (or all rows if not specified),
    removes urls and saves result in column 'text',
    creates new column with word count from column 'text',
    creates new column 'chunk_index' with 0s,
    Returns: prepared dataframe
    """
    df = df.dropna()
    if n_rows is None:
        n_rows = len(df)
    df = df.sample(n_rows).copy().reset_index(drop=True)   # n random rows
    df['text']=df[col_with_text].apply(remove_urls)
    df['text_word_count'] = df['text'].str.split().str.len()
    df['chunk_index'] = 0
    return df


def chunk_text_by_tokens(text: str, tokenizer, max_tokens = 256, overlap=10) -> List[str]:
    """
    Split long text into smaller chunks by the number of tokens with optional overlap.
    Returns: A list of text chunks (with 1 element if len(tokens) < max_tokens).
    """
  
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunk = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk)
            start = end - overlap
    else:
        chunks = [tokenizer.convert_tokens_to_string(tokens)]
    return chunks



def add_text_chunks_to_df(df: DataFrame, id_column_name: str, tokenizer,  max_tokens = 256, overlap = 10) -> DataFrame:
    """
    For each row in df, create a list of text chunks,
    add first chunk in list to the original row,
    add additional chunks from list on to new df rows where all information is copied from original row except for chunk_index which corresponds to element number in list of chunks
    sort dataframe on a column unique for each original text and chunk_index,
    remove rows where text_chunk has content
    Retrun dataframe with text_chunks
    """
    # for each row in df return chunks = [chunk, chunk, ...]
    for index, row in df.iterrows():
        chunks = chunk_text_by_tokens(row['text'], tokenizer=tokenizer)

        for chunk_index, chunk in enumerate(chunks):
            # 1st chunk in chunks -> add chunk to new column 'text_chunk'
            if chunk_index == 0:
                df.loc[index, 'text_chunk'] = chunk
            # subsequent chunk, chunk, ... in chunks -> create a df which is a copy of the current row, updated it with chunk_index and add text_chunk -> concatinate that 1 row df to original df
            else:
                df_new_row = pd.DataFrame(row).T
                df_new_row['chunk_index'] = chunk_index
                df_new_row['text_chunk'] = chunk
                df = pd.concat([df, df_new_row], ignore_index = True)

    df = df.sort_values([id_column_name, 'chunk_index'])
    df = df[df['text_chunk'].str.len() > 0].reset_index(drop=True) # keep only rows where text_chunk has content and reset index
    return df