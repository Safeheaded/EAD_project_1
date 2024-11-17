import os
from typing import LiteralString
import matplotlib.pyplot as plt
import numpy as np

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

def zad5(df: pd.DataFrame):
    counted_births = df.groupby(["Year"]).agg({"Count": "sum"}).reset_index()
    x = list(set(df['Year']))

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(x, counted_births['Count'])
    axes[0].legend(['Births per year'])


    gender_counts = df.groupby(["Year", "Gender"]).agg({"Count": "sum"})

    female_counts = gender_counts.loc[(slice(None), 'F'), 'Count'].reset_index()
    male_counts = gender_counts.loc[(slice(None), 'M'), 'Count'].reset_index()
    ratio = female_counts.loc[:, 'Count'] / male_counts.loc[:, 'Count']

    axes[1].plot(x, ratio)

    highest_diff_year = x[np.argmax(ratio)] if x[np.argmax(ratio)] > x[np.argmin(ratio)] else x[np.argmin(ratio)]

    closest_to_one_index = (ratio - 1).abs().idxmin()
    closest_to_one_value = ratio.loc[closest_to_one_index]
    closest_to_one_year = x[closest_to_one_index]

    axes[1].plot(highest_diff_year, ratio[np.argmax(ratio)], 'ro')
    axes[1].plot(closest_to_one_year, closest_to_one_value, 'bo')
    axes[1].legend(['Female to male ratio', 'Highest difference', 'Lowest difference'])
    plt.show()

    print(f'Year with the highest difference in births: {highest_diff_year}')
    print(f"Value closest to 1: {closest_to_one_value} at index {closest_to_one_index}")

def zad6(df: pd.DataFrame):
    counted_births = df.groupby(["Year"]).agg({"Count": "sum"})
    counted_births = counted_births.rename(columns={"Count": "General_count"})
    top_1000_per_year_gender = df.groupby(['Year', 'Gender']).apply(
        lambda x: x.nlargest(1000, 'Count')
    ).reset_index(drop=True)
    top_1000_per_year_gender = top_1000_per_year_gender.merge(counted_births, on=['Year'])

    top_1000_per_year_gender['ratio'] = top_1000_per_year_gender['Count'] / top_1000_per_year_gender['General_count']

    ag = top_1000_per_year_gender.groupby(["Name", "Gender"]).agg({"ratio": "mean"}).reset_index()
    top_1000_male_names = ag[ag["Gender"] == "M"].nlargest(1000, 'ratio')
    top_1000_female_names = ag[ag["Gender"] == "F"].nlargest(1000, 'ratio')
    return top_1000_male_names, top_1000_female_names, top_1000_per_year_gender

def zad7(df: pd.DataFrame):
    top_males, top_females, all_top_1000 = zad6(df)
    male_name = 'John'
    female_name = top_females.iloc[0, 0]

    x = list(set(df['Year']))

    data = df.groupby(["Year", "Name", "Gender"]).agg({"Count": "sum"}).reset_index()

    male_data = data.where(data["Gender"] == "M").dropna()
    female_data = data.where(data["Gender"] == "F").dropna()

    female_name_data = female_data.where(data["Name"] == female_name).dropna().loc[:, ["Count", "Year"]].reset_index()
    male_name_data = male_data.where(data["Name"] == male_name).dropna().loc[:, ["Count", "Year"]].reset_index()

    all_top_1000 = all_top_1000.reset_index()

    female_popularity = all_top_1000.where(all_top_1000["Gender"] == "F")[['Name', 'ratio', "Year", "Gender"]].where(all_top_1000["Name"] == female_name).dropna().loc[:, ["ratio", "Year"]]

    male_popularity = all_top_1000.where(all_top_1000["Gender"] == "M")[['Name', 'ratio', "Year", "Gender"]].where(all_top_1000["Name"] == male_name).dropna().loc[:, ["ratio", "Year"]]

    years_to_print = [1934, 1980, 2022]
    print(f"Counts for male name {male_name}:")
    print(female_name_data.loc[female_name_data["Year"].isin(years_to_print), ["Count", "Year"]])
    print(f"Counts for female name {female_name}:")
    print(male_name_data.loc[male_name_data["Year"].isin(years_to_print), ["Count", "Year"]])

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Year')
    ax1.set_ylabel(f'Count', color='tab:blue')
    ax1.plot(male_name_data["Year"], male_name_data["Count"], color='tab:blue')
    ax1.tick_params(axis='y')
    ax1.plot(female_name_data["Year"], female_name_data["Count"], color='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Popularity', color='tab:green')
    ax2.plot(male_popularity["Year"], male_popularity["ratio"], color='tab:green')
    ax2.tick_params(axis='y')
    ax2.plot(female_popularity["Year"], female_popularity["ratio"], color='tab:orange')
    ax1.legend([
    f"Count of name {male_name}", f"Count of name {female_name}", f"Popularity of name {male_name}", f"Popularity of name {female_name}"
    ])
    ax2.legend([
    f"Popularity of name {male_name}", f"Popularity of name {female_name}"
    ], loc='upper left')

    fig.tight_layout()
    plt.show()

def main():
    df = get_txt_dataset()
    zad7(df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
