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
    dataframes = []
    for path in paths:
        df = pd.read_csv(path, header=None, names=column_names)
        year = os.path.basename(path).split('yob')[1].split('.')[0]
        df['Year'] = int(year)
        dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

def zad2(df: pd.DataFrame):
    unique_names = set(df.loc[:, "Name"])
    print(f"Number of unique names: {len(unique_names)}")

def zad3(df: pd.DataFrame):
    unique_female_names = set(df.loc[df["Gender"] == "F", "Name"])
    unique_male_names = set(df.loc[df["Gender"] == "M", "Name"])
    print(f"Number of unique male names: {len(unique_male_names)}")
    print(f"Number of unique female names: {len(unique_female_names)}")

def zad4(df: pd.DataFrame):
    counted_births = df.groupby(["Year", "Gender"]).agg({"Count": "sum"})
    counted_births = counted_births.rename(columns={"Count": "General_count"})

    df = df.merge(counted_births, on=["Year", "Gender"])
    df['female_frequency'] = (df['Count'] / df['General_count']).mask(df['Gender'] == 'M', 0)
    df['male_frequency'] = (df['Count'] / df['General_count']).mask(df['Gender'] == 'F', 0)

    print(df)

def main():
    df = get_txt_dataset()
    zad4(df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
