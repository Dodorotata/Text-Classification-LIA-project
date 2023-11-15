from datetime import datetime
from pathlib import Path
from plotly.graph_objs import Figure
import pandas as pd
import json

def create_experiment_folder(base_path: Path, file_name: str) -> Path:
    """
    Create an experiment folder based on current timestamp and provided file name.
    - base_path (Path): The base directory where the experiment folder will be created.
    - file_name (str): The name to incorporate into the experiment folder name.
    """
    results_folder = f"results_{file_name}"
    experiment_folder = f"exp_{datetime.now().strftime('%y%m%d_%H%M%S')}"
    folder_path = base_path / results_folder / experiment_folder
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path



def save_parameters(params: dict, folder_path: Path) -> None:
    """
    Save experiment parameters to a JSON file in the specified folder.
    """
    params_file_path = folder_path / 'parameters.json'
    with open(params_file_path, 'w') as json_file:
        json.dump(params, json_file)


def save_dataframe(df: pd.DataFrame, folder_path: Path) -> None:
    """
    Save a DataFrame to a TSV file in the specified folder.
    """
    tsv_path = folder_path / 'full_analysis_data.tsv'
    df.to_csv(tsv_path, index=False)


def save_plot(fig: Figure, folder_path: Path) -> None:
    """
    Save a Plotly figure to an HTML file in the specified folder.
    """
    fig_path = folder_path / 'fig_clustered_text_data.html'
    fig.write_html(fig_path)