import pandas as pd
import os


def read_csv(src_path):
    return pd.read_csv(src_path)


def write_csv(content, dest_folder: str, file_name: str, columns: list = None, with_index: bool = False):
    sub = pd.DataFrame(content, columns=None)
    sub.to_csv(os.path.join(dest_folder, file_name), index=with_index, encoding="utf-8")
