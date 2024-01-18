import os
from src.mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import spacy
import re
import pickle
from tmtoolkit.corpus import Corpus
from tmtoolkit.corpus import doc_labels
from tmtoolkit.corpus import (lemmatize, filter_for_pos, to_lowercase,
    remove_punctuation, filter_clean_tokens, remove_common_tokens,
    remove_uncommon_tokens,
    tokens_table)
from tmtoolkit.corpus import vocabulary
from tmtoolkit.bow.dtm import create_sparse_dtm
from tmtoolkit.corpus import dtm
from tmtoolkit.bow.bow_stats import tfidf, tf_proportions, idf
from src.mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self,config: DataTransformationConfig):
        self.config = config
    

    def train_test_spliting(self):
        # chargement des fichiers .txt en un corpus
        nlp = spacy.load("fr_core_news_lg")
        corpus = Corpus.from_folder("data", language_model="fr_core_news_lg")
        # on extrait les noms des fichiers pour créer un dictionnaire contenant la classe des mails (spam ou not-spam)
        labels = doc_labels(corpus)
        classes = []
        spam_pattern = re.compile(r'^spmsg')
        for item in labels:
            if spam_pattern.match(item):
                classes.append("spam")
            else:
                classes.append("not-spam")
        classes_dico = dict(zip(labels, classes))

        # on prétraite notre corpus
        train_corpus_l = lemmatize(corpus, inplace=False)
        to_lowercase(train_corpus_l)
        filter_clean_tokens(train_corpus_l, remove_shorter_than=2, remove_longer_than=20, remove_numbers = True)
        remove_punctuation(train_corpus_l)
        remove_common_tokens(train_corpus_l, df_threshold=0.9)
        remove_uncommon_tokens(train_corpus_l, df_threshold=0.005)
        # on extrait le vocabulaire
        vocab_train = np.array(vocabulary(train_corpus_l))

        #on met notre corpus en une représentation tf-idf
        mat_dtm_train = dtm(train_corpus_l)
        tfidf_mat_train = tfidf(mat_dtm_train, tf_func=tf_proportions, idf_func=idf)
        tfidf_mat_train.to_csv(os.path.join(self.config.root_dir,'data.csv'))
        # on sauvegarde la tf-idf dans un fichier pickle
        with open(os.path.join(self.config.root_dir,'data.pkl'), 'wb') as file:
            pickle.dump(tfidf_mat_train, file)
        #On tient compte du fait que dtm change l'ordre des documents. Les targets à prédire sont donc modifiées pour suivre l'ordre des matrices
        new_train_class = [classes_dico[id_doc] for id_doc in doc_labels(train_corpus_l)]
        # on sauvegarde nos classes en csv
        new_train_class.to_csv(os.path.join(self.config.root_dir,'classes.csv'),index=False)
    
        