import os
from cProfile import label
from typing import LiteralString
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from numpy.ma.core import argmax


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

def zad8(df: pd.DataFrame):
    top_males, top_females, all_top_1000 = zad6(df)
    counted_births = df.groupby(["Year", "Name", "Gender"]).agg({"Count": "sum"})
    counted_births["total_yearly_population"] = counted_births.groupby(["Year", "Gender"])["Count"].transform('sum')

    top_male_names = top_males['Name'].tolist()
    top_female_names = top_females['Name'].tolist()

    counted_births_females = counted_births.loc[(slice(None), slice(None), "F"), :]
    counted_births_males = counted_births.loc[(slice(None), slice(None), "M"), :]

    # Musisz to przefiltrować oddzielnie dla bab i chłopów
    counted_births_females_filtered = counted_births_females[counted_births_females.index.get_level_values('Name').isin(top_female_names)]
    counted_births_males_filtered = counted_births_males[counted_births_males.index.get_level_values('Name').isin(top_male_names)]

    counted_births_females_filtered["population_percentage"] = counted_births_females_filtered["Count"] / counted_births_females_filtered["total_yearly_population"]
    counted_births_males_filtered["population_percentage"] = counted_births_males_filtered["Count"] / counted_births_males_filtered["total_yearly_population"]

    females_aggregated = counted_births_females_filtered.groupby("Year").agg({"population_percentage": "sum"})
    males_aggregated = counted_births_males_filtered.groupby("Year").agg({"population_percentage": "sum"})

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(females_aggregated.index, females_aggregated["population_percentage"], label="Females")
    plt.plot(males_aggregated.index, males_aggregated["population_percentage"], label="Males")
    plt.xlabel("Year")
    plt.ylabel("Population Percentage")
    plt.title("Population Percentage of Top Names by Year")
    print(females_aggregated)
    females_aggregated["diff"] = abs(females_aggregated.loc[:, "population_percentage"] - males_aggregated.loc[:,"population_percentage"])
    biggest_diff_year = females_aggregated.reset_index().iloc[argmax(females_aggregated["diff"]), 0]

    min_val = min(females_aggregated["population_percentage"])
    min_val_male = min(males_aggregated["population_percentage"])
    min_min = min(min_val, min_val_male)

    # plt.axvline(x=biggest_diff_year, color='r', linestyle='--')
    plt.scatter(biggest_diff_year, females_aggregated.loc[biggest_diff_year, "population_percentage"], color='r', label="Biggest difference year")
    plt.scatter(biggest_diff_year, males_aggregated.loc[biggest_diff_year, "population_percentage"], color='r')
    plt.vlines(x=biggest_diff_year, ymin=min_min, ymax=females_aggregated.loc[biggest_diff_year, "population_percentage"], color='r', linestyle='--')
    plt.vlines(x=biggest_diff_year, ymin=min_min, ymax=males_aggregated.loc[biggest_diff_year, "population_percentage"], color='r', linestyle='--')

    plt.legend()
    plt.show()

    print(f"Year with the biggest difference in population percentage: {biggest_diff_year}")

def zad9(df: pd.DataFrame):
    df["last_letter"] = df["Name"].str[-1]
    # print(df[df["Year"] == 1910])

    # df["last_letter"] = pd.factorize(df["last_letter"])[0]

    df_total = df.groupby(["Year", "Gender"]).agg({"Count": "sum"}).rename(columns={"Count": "total_count"}).reset_index()
    df = df.groupby(["Year", "Gender", "last_letter"]).agg({"Count": "sum"}).rename(columns={"Count": "letter_count"}).reset_index()
    df = df.merge(df_total, on=["Year", "Gender"])
    df["ratio"] = df["letter_count"] / df["total_count"]

    # print(df)
    # print(pd.crosstab(index=df["last_letter"], columns=df["letter_count"], normalize="columns"))

    def parse_yearly_data(year: int) -> pd.DataFrame:
        data = df.loc[df["Year"] == year]
        data = data.loc[data["Gender"] == "M"]
        return data

    year_1910 = parse_yearly_data(1910)
    year_1970 = parse_yearly_data(1970)
    year_2023 = parse_yearly_data(2023)

    # print(year_1910)

    all_letters = set(year_1910["last_letter"]).union(set(year_1970["last_letter"]), set(year_2023["last_letter"]))
    all_letters = sorted(all_letters)
    year_1910 = year_1910.set_index("last_letter").reindex(all_letters, fill_value=0).reset_index()
    year_1970 = year_1970.set_index("last_letter").reindex(all_letters, fill_value=0).reset_index()
    year_2023 = year_2023.set_index("last_letter").reindex(all_letters, fill_value=0).reset_index()

    plt.figure(figsize=(15, 8))
    bar_width = 0.25

    r1 = range(len(year_1910))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    plt.bar(r1, year_1910["ratio"], color='b', width=bar_width, edgecolor='grey', label='1910')
    plt.bar(r2, year_1970["ratio"], color='g', width=bar_width, edgecolor='grey', label='1970')
    plt.bar(r3, year_2023["ratio"], color='r', width=bar_width, edgecolor='grey', label='2023')

    plt.xlabel('Last Letter', fontweight='bold')
    plt.ylabel('Ratio', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(year_1910))], year_1910["last_letter"])

    plt.title("Ratio of Last Letters for Male Names in 1910, 1970, and 2023")
    plt.legend()
    plt.show()
    # print(year_1910.index)
    #
    # year_1910 = pd.crosstab(index=[year_1910["last_letter"], year_1910["Gender"]], columns=[year_1910["letter_count"]], values=year_1910["letter_count"], aggfunc='sum', normalize=True)
    #
    # year_1910.loc[(slice(None), "M"), :].plot(kind="bar", stacked=True, figsize=(15, 7))

    # print(df)

    diff = year_2023["ratio"] - year_1910["ratio"]
    diff = diff.abs()
    max_diff = diff.idxmax()
    print(f"Biggest difference in ratio between 1910 and 2023: {year_2023.iloc[max_diff, 0]} with value {diff[max_diff]}")
    min_diff = diff.idxmin()
    print(f"Smallest difference in ratio between 1910 and 2023: {year_2023.iloc[min_diff, 0]} with value {diff[min_diff]}")
    year_2023["diff"] = diff
    year_2023 = year_2023.sort_values(by="diff", ascending=False)

    top_difference_letters = year_2023.head(3)["last_letter"]
    top_letters_pop = df[df["Gender"] == "M"].where(df["last_letter"].isin(top_difference_letters)).dropna().groupby(["Year", "last_letter", "ratio"]).agg({"ratio": "sum"}).rename(columns={"ratio": "fin_ratio"})
    x = list(set(top_letters_pop.index.get_level_values("Year")))

    plt.figure(figsize=(10, 6))
    for letter in top_difference_letters:
        plt.plot(x, top_letters_pop.loc[(slice(None), letter, slice(None)),"fin_ratio"], label=f"Ratio of {letter}")
    plt.xlabel("Year")
    plt.ylabel("Ratio")
    plt.title("Popularity of Top Letters by Year")
    plt.legend()
    plt.show()

def main():
    df = get_txt_dataset()
    zad9(df)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
