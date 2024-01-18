import os
import urllib.request as request
import zipfile
import spacy
from tmtoolkit.corpus import Corpus

from src.mlProject import logger
from src.mlProject.utils.common import get_size
from src.mlProject.entity.config_entity import DataIngestionConfig


from pathlib import Path

class DataIngestion:
    def __init__(self, config:DataIngestionConfig):
        self.config = config

    def load_data_files(self):
        nlp = spacy.load("fr_core_news_lg")
        corpus = Corpus.from_folder(self.config.local_data_file, language_model="fr_core_news_lg")
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename,headers =request.urlretrieve(
                url = self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f'{filename} download! with following info: \n{headers}')
        else:
            logger.info(f'File already exists of size : {get_size(Path(self.config.local_data_file))}')
    
    def extract_zip_file(self):
        unzip_file=self.config.unzip_dir
        try:
            os.makedirs(unzip_file, exist_ok=True)
        except FileExistsError:
            pass
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_file)