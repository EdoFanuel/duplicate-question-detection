import pandas as pd


def read_csv(src_path):
    return pd.read_csv(src_path)


def write_csv(content, headings, dest_path):
    sub = pd.DataFrame(data=content, columns=headings)
    sub.to_csv(dest_path, index=False)
