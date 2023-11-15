import plotly.express as px
import pandas as pd
from plotly.graph_objs import Figure


def create_topic_cluster_scatter(df: pd.DataFrame, category: str) -> Figure:
    """
    Create and combine two scatter plots:
        (1) clustered data points (non-negative cluster labels)
        (2) outliers(cluster label -1)
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing data points with cluster labels.
    - category (str): The name of the column to use for coloring data points.
    """
    
    outliers = df.loc[df.cluster_label == -1, :]
    clustered = df.loc[df.cluster_label != -1, :]
    
    # scatter plot for clustered data points
    fig = px.scatter(clustered, x='umap_x', y='umap_y', color= category, hover_data = {
        'umap_x': False,
        'umap_y': False,
        'top_words': True,
        'cluster_label': True
    })
    
    fig.update_traces(hovertemplate='Top words: %{customdata[0]}<br>Cluster=%{marker.color}')

     # scatter plot for outliers
    outliers_scatter = px.scatter(outliers, x='umap_x', y='umap_y', hover_data = {
        'umap_x': False,
        'umap_y': False,
        'top_words': False,
        'cluster_label': True
    } )
    outliers_scatter.update_traces(marker=dict(color='lightgray', size = 5), text = 'outliers', hovertemplate='Unclassified')

    # add the outlier scatter plot to fig with clusters
    fig.add_trace(outliers_scatter.data[0])
    return fig



def custom_scatter_layout(fig: Figure, plot_title: str, x_title: str, y_title: str) -> Figure:
    """
    Customizes the layout of a Plotly scatter plot.
    """
    fig.update_layout(
        plot_bgcolor = 'white',
        title=plot_title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=700,
        width=1000
    )

    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='white'
    )

    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='white'
    )

    return fig
