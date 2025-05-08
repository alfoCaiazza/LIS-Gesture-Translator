import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Percorso assoluto alla cartella `src/data`, relativo allo script
base_dir = os.path.dirname(os.path.dirname(__file__))  # va su di un livello
data_dir = os.path.join(base_dir, 'data')

# Crea la directory se non esiste
os.makedirs(data_dir, exist_ok=True)

# Inizializza l'API
api = KaggleApi()
api.authenticate()

# Scarica e decomprimi il dataset nella cartella `src/data`
api.dataset_download_files('nicholasnisopoli/lisdataset', path=data_dir, unzip=True)

print(f"Dataset donwloaded in: {data_dir}")
