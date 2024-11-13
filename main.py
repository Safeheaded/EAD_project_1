import os
from typing import LiteralString

import pandas as pd

def get_file_paths(directory: str) -> list[str]:
    file_paths: list[str] = list[str]()
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def get_txt_dataset() -> pd.DataFrame:
    paths = get_file_paths("./data")

    column_names = ["Name", "Gender", "Count"]
    dataframes = [pd.read_csv(path, header=None, names=column_names) for path in paths]
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def zad1(df: pd.DataFrame):
    unique_names = set(df.loc[:, "Name"])
    print(f"Number of unique names: {len(unique_names)}")

def zad2(df: pd.DataFrame):
    unique_female_names = set(df.loc[df["Gender"] == "F", "Name"])
    unique_male_names = set(df.loc[df["Gender"] == "M", "Name"])
    print(f"Number of unique male names: {len(unique_male_names)}")
    print(f"Number of unique female names: {len(unique_female_names)}")

def main():
    df = get_txt_dataset()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
