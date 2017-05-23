import pandas as pd
import numpy as np
import os.path as osp
from gensim import corpora
from main import function as funct
from datetime import datetime as dt


def read_csv(src_path: str, data_types: dict = None) -> pd.DataFrame:
    return pd.read_csv(src_path, dtype=data_types)


def write_csv(content: list, dest_folder: str, file_name: str, columns: list = None, with_index: bool = False) -> None:
    sub = pd.DataFrame(content, columns=columns)
    sub.to_csv(osp.join(dest_folder, file_name), index=with_index, encoding="utf-8")


def reload_dictionary(file_path: str, corpus: list) -> corpora.Dictionary:
    if osp.isfile(file_path):
        return corpora.Dictionary.load(file_path)
    else:
        start_time = dt.now()
        print("Generating Token Dictionary")
        dictionary = funct.generate_dictionary(corpus)
        dictionary.save(file_path)
        elapsed_time = dt.now() - start_time
        print("Dictionary generated in {0} seconds, located at {1}".format(elapsed_time.total_seconds(), file_path))
        return dictionary
