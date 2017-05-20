import pandas as pd


def read_csv(src_path):
    return pd.read_csv(src_path)


def write_csv(content, headings: list, dest_path: str):
    sub = pd.DataFrame(data=content, columns=headings)
    sub.to_csv(dest_path, index=False)
