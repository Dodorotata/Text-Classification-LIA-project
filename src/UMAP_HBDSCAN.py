# %% [markdown]
# Notebook details:
# * handles all or slice from dataset (random choice of rows)
# * text preprocessing: remove all urls, remove empty chunk rows
# * text is devided in chunks to avoid truncation by BERT model
# * no parametertunig of any of the models
# * dimensionality reduction is done with UMAP
# * clustering with HBDSCAN
# * labeling is done by counting the most common words in each cluster after lemmatization
# * results and parameters for each run are saved in folder
# --------

# %%
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from typing import List
import numpy as np
import torch
from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from umap import UMAP
from hdbscan import HDBSCAN
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
from datetime import datetime
import json
import webbrowser

from tools.text.filtering import remove_urls
from preprocessing import read_raw_data_to_df, prepare_df, add_text_chunks_to_df
from cluster_labeling import c_tf_idf, extract_top_n_words_per_topic, extract_cluster_sizes
from visualisation import create_topic_cluster_scatter, custom_scatter_layout
from save_results import create_experiment_folder, save_plot, save_dataframe, save_parameters

# Create df from data
file_name = 'vnnforum_small.tsv'
df = read_raw_data_to_df(file_name)
df = df
col_containing_text = 'text'
n_rows = 1000
df = prepare_df(df, col_containing_text, n_rows)


# Save text in chunks short enough for model to handle and add to df
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
id_col = 'article_id'
df = add_text_chunks_to_df(df = df, tokenizer=tokenizer, id_column_name = id_col)


# Calculate embeddings, add to df and save in datawarehouse
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_list = df["text_chunk"].to_list()
embeddings = model.encode(chunk_list, show_progress_bar=True, normalize_embeddings=True)
df['chunk_embedding'] = list(embeddings)

datawarehouse_folder = Path(__file__).parents[1] / 'datawarehouse'
datawarehouse_folder.mkdir(parents=True, exist_ok=True)
df.to_csv(f'{datawarehouse_folder}/{file_name}_chunked_embeddings.tsv', sep="\t", index=False)


# Create folder for current experiment
base_path = Path(__file__).parents[1]
file_name = file_name
exp_folder_path = create_experiment_folder(base_path, file_name)

# Reduce embedings dimensionality with UMAP and cluster with HBDSCAN, add cluster label to df
umap_params = {
    'n_neighbors':20,
    'n_components':8,
    'min_dist':0.05, 
    'metric':'cosine'
}

umap_embeddings = UMAP(**umap_params).fit_transform(embeddings)

hbdscan_params = {
    'min_cluster_size':20,
    'metric':'euclidean',
    #'min_samples':40,
    'gen_min_span_tree':True,
    'prediction_data':True,       
    'cluster_selection_method':'eom'
}

cluster = HDBSCAN(**hbdscan_params).fit(umap_embeddings)

df['cluster_label'] = cluster.labels_


# Lematize and calculate most frequent words for each cluster
# NOTE: words like jews, jewish are not lemmatized to jew, which should be further addressed with SpaCy
clustered_docs_df = df.groupby(['cluster_label'], as_index = False).agg({'text_chunk': ' '.join})

nlp = spacy.load("en_core_web_sm")
clustered_docs_df['lemmatized_text_chunk'] = clustered_docs_df['text_chunk'].apply(lambda text: ' '.join(token.lemma_ for token in nlp(text)))
tf_idf, count = c_tf_idf(clustered_docs_df['lemmatized_text_chunk'].values, m=len(chunk_list))
top_n_words = extract_top_n_words_per_topic(tf_idf, count, clustered_docs_df, n=10)
cluster_words_df = extract_cluster_sizes(df)
cluster_words_df['top_words'] = cluster_words_df['cluster_label'].apply(lambda label: [word for word, _ in top_n_words[label]])
print(cluster_words_df)


# Prepare for visualization in 2D and gather all in df
umap_params_2D = umap_params.copy()
umap_params_2D['n_components'] = 2
umap_embeddings_2D = UMAP(**umap_params_2D).fit_transform(embeddings)

df['umap_x'] = umap_embeddings_2D[:, 0]
df['umap_y'] = umap_embeddings_2D[:, 1]

df = df.merge(cluster_words_df, on='cluster_label').drop('cluster_size', axis=1) # *1)

# Visualize topic clusters, save plot, open in browser
fig = create_topic_cluster_scatter(df = df, category = 'cluster_label')
fig = custom_scatter_layout(fig = fig, plot_title = 'VNN blogg grouped by cluster', x_title = 'umap_x', y_title = 'umap_y')
save_plot(fig, exp_folder_path)

cluster_plot_path = exp_folder_path / 'fig_clustered_text_data.html'
webbrowser.open(str(cluster_plot_path), new=2)  # 'new=2' opens in a new tab or window

# Save df with all data
save_dataframe(df, exp_folder_path)

# Save paramteres
params = {
    'raw_data_file': file_name,
    'tokenizer_for_creating_chunks': str(tokenizer),
    'embeddings_model': str(model),
    'umap_params': umap_params,
    'hdbscan_params': hbdscan_params
}

save_parameters(params, exp_folder_path)


