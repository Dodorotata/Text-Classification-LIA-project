import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Tuple, Union, Dict
from pandas import DataFrame
from numpy import ndarray
from scipy.sparse import csr_matrix



def c_tf_idf(documents: List[str], m: int, ngram_range: Tuple[int, int] = (1, 1)) -> Tuple[ndarray, CountVectorizer]:
    """
    Calculate the c-TF-IDF (class-based Term Frequency-Inverse Document Frequency) for a list of documents.

    Parameters:
    - documents: A list of text documents for which c-TF-IDF will be calculated.
    - m: The total number of documents in the corpus.
    - ngram_range: A tuple specifying the range of n-grams to consider. Default is (1, 1), which means
      only single words are considered since it stands for (Lower Bound, Upper Bound)

    Returns:
    - tf_idf (ndarray): An array containing the c-TF-IDF values for each term in the documents.
    - count (CountVectorizer): The CountVectorizer object used to fit and transform the documents.
    """
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf: csr_matrix, count: CountVectorizer, df: pd.DataFrame, n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
    """
    Extracts the top n words per topic based on TF-IDF scores.

    Parameters:
    - tf_idf (scipy.sparse.csr_matrix): TF-IDF matrix with documents as rows and words as columns.
    - count (CountVectorizer): CountVectorizer object used for tokenization.
    - df (pandas.DataFrame): DataFrame containing cluster labels for each document.
    - n (int, optional): Number of top words to extract for each topic. Default is 10.

    Returns:
    - top_n_words (dict): A dictionary where keys are cluster labels and values are lists of tuples.
      Each tuple contains a word and its corresponding TF-IDF score, representing the top n words for that topic.
    """
    words = count.get_feature_names_out()
    labels = list(df.cluster_label)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def extract_cluster_sizes(df: DataFrame) -> DataFrame:
    """
    Extracts the sizes of clusters based on the count of documents in each cluster.

    Parameters:
    - df (pd.DataFrame): DataFrame containing cluster labels and document information.

    Returns:
    - cluster_sizes (pd.DataFrame): DataFrame with two columns, 'cluster_label' and 'size',
      representing the cluster labels and the number of documents in each cluster.
      The DataFrame is sorted in descending order by cluster size.
    """
    cluster_sizes = (df.groupby(['cluster_label'])
                     .text_chunk
                     .count()
                     .reset_index()
                     .rename({"text_chunk": "cluster_size"}, axis='columns')
                     .sort_values("cluster_size", ascending=False))
    return cluster_sizes

