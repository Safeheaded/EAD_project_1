import os
import sqlite3
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
            if file.endswith('.txt'):
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
    df['frequency_female'] = (df['Count'] / df['General_count']).mask(df['Gender'] == 'M', 0)
    df['frequency_male'] = (df['Count'] / df['General_count']).mask(df['Gender'] == 'F', 0)

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
    print(f'Year with the smallest difference in births: {closest_to_one_year}')

def zad6(df: pd.DataFrame, names_num=1000, split_by_gender=True, should_display=True):
    counts_columns = ["Year", "Gender"] if split_by_gender else ["Year"]
    gender_counts = df.groupby(counts_columns).agg({"Count": "sum"}).rename(columns={"Count": "Total_count"})
    # gender_counts = gender_counts.reset_index()

    # female_counts = gender_counts.loc[(slice(None), 'F'), 'Count'].reset_index()
    # male_counts = gender_counts.loc[(slice(None), 'M'), 'Count'].reset_index()
    counted_columns = ["Year", "Gender", "Name"] if split_by_gender else ["Year", "Name"]
    counted = df.groupby(counted_columns).agg({"Count": "sum"}).reset_index()
    counted = counted

    counted = gender_counts.merge(counted, on=counts_columns)
    # print(counted)

    counted["popularity"] = counted["Count"] / counted["Total_count"]

    with_popularity = counted.copy()

    counted_second_columns = ["Gender", "Name"] if split_by_gender else ["Name"]
    counted = counted.groupby(counted_second_columns).agg({"popularity": "sum"}).reset_index()


    females = counted[counted["Gender"] == "F"] if split_by_gender else counted
    males = counted[counted["Gender"] == "M"] if split_by_gender else counted

    top_1000_females = females.sort_values(by="popularity", ascending=False).head(names_num)
    top_1000_males = males.sort_values(by="popularity", ascending=False).head(names_num)

    if should_display:
        print(f"Top {names_num} females:")
        print(top_1000_females)
        print(f"Top {names_num} males:")
        print(top_1000_males)

    return top_1000_males, top_1000_females, with_popularity

def zad7(df: pd.DataFrame):
    top_males, top_females, all_top_1000 = zad6(df, should_display=False)
    male_name = 'John'
    female_name = top_females.iloc[0, 1]

    data = df.groupby(["Year", "Name", "Gender"]).agg({"Count": "sum"}).reset_index()

    male_data = data.where(data["Gender"] == "M").dropna()
    female_data = data.where(data["Gender"] == "F").dropna()

    female_name_data = female_data.where(data["Name"] == female_name).dropna().loc[:, ["Count", "Year"]].reset_index()
    male_name_data = male_data.where(data["Name"] == male_name).dropna().loc[:, ["Count", "Year"]].reset_index()

    all_top_1000 = all_top_1000

    female_popularity = all_top_1000.where(all_top_1000["Gender"] == "F")[['Name', 'popularity', "Year", "Gender"]].where(all_top_1000["Name"] == female_name).dropna().loc[:, ["popularity", "Year"]]

    male_popularity = all_top_1000.where(all_top_1000["Gender"] == "M")[['Name', 'popularity', "Year", "Gender"]].where(all_top_1000["Name"] == male_name).dropna().loc[:, ["popularity", "Year"]]

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
    ax2.plot(male_popularity["Year"], male_popularity["popularity"], color='tab:green')
    ax2.tick_params(axis='y')
    ax2.plot(female_popularity["Year"], female_popularity["popularity"], color='tab:orange')
    ax1.legend([
    f"Count of name {male_name}", f"Count of name {female_name}", f"Popularity of name {male_name}", f"Popularity of name {female_name}"
    ])
    ax2.legend([
    f"Popularity of name {male_name}", f"Popularity of name {female_name}"
    ], loc='upper left')

    fig.tight_layout()
    plt.show()

def zad8(df: pd.DataFrame, names_num=1000, split_by_gender=True, should_display=True):
    top_males, top_females, all_top_1000 = zad6(df, names_num, split_by_gender, should_display=False)
    counted_births_columns = ["Year", "Name", "Gender"] if split_by_gender else ["Year", "Name"]
    counted_births = df.groupby(counted_births_columns).agg({"Count": "sum"})
    total_columns = ["Year", "Gender"] if split_by_gender else ["Year"]
    counted_births["total_yearly_population"] = counted_births.groupby(total_columns)["Count"].transform('sum')

    top_male_names = top_males['Name'].tolist()
    top_female_names = top_females['Name'].tolist()

    counted_births_females = counted_births.loc[(slice(None), slice(None), "F"), :] if split_by_gender else counted_births
    counted_births_males = counted_births.loc[(slice(None), slice(None), "M"), :] if split_by_gender else counted_births

    counted_births_females_filtered = counted_births_females[counted_births_females.index.get_level_values('Name').isin(top_female_names)]
    counted_births_males_filtered = counted_births_males[counted_births_males.index.get_level_values('Name').isin(top_male_names)]

    counted_births_females_filtered.loc[:,"population_percentage"] = counted_births_females_filtered["Count"] / counted_births_females_filtered["total_yearly_population"]
    counted_births_males_filtered.loc[:,"population_percentage"] = counted_births_males_filtered["Count"] / counted_births_males_filtered["total_yearly_population"]

    females_aggregated = counted_births_females_filtered.groupby("Year").agg({"population_percentage": "sum"})
    males_aggregated = counted_births_males_filtered.groupby("Year").agg({"population_percentage": "sum"})

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(females_aggregated.index, females_aggregated["population_percentage"], label="Females" if split_by_gender else "Percentage of top names")
    if split_by_gender:
        plt.plot(males_aggregated.index, males_aggregated["population_percentage"], label="Males")
    plt.xlabel("Year")
    plt.ylabel("Population Percentage")
    plt.title("Population Percentage of Top Names by Year")
    if split_by_gender:
        females_aggregated["diff"] = abs(
            females_aggregated.loc[:, "population_percentage"] - males_aggregated.loc[:, "population_percentage"])
        biggest_diff_year = females_aggregated.reset_index().iloc[argmax(females_aggregated["diff"]), 0]

        min_val = min(females_aggregated["population_percentage"])
        min_val_male = min(males_aggregated["population_percentage"])
        min_min = min(min_val, min_val_male)

        # plt.axvline(x=biggest_diff_year, color='r', linestyle='--')
        plt.scatter(biggest_diff_year, females_aggregated.loc[biggest_diff_year, "population_percentage"], color='r',
                    label="Biggest difference year")
        plt.scatter(biggest_diff_year, males_aggregated.loc[biggest_diff_year, "population_percentage"], color='r')
        plt.vlines(x=biggest_diff_year, ymin=min_min,
                   ymax=females_aggregated.loc[biggest_diff_year, "population_percentage"], color='r', linestyle='--')
        plt.vlines(x=biggest_diff_year, ymin=min_min,
                   ymax=males_aggregated.loc[biggest_diff_year, "population_percentage"], color='r', linestyle='--')
        print(f"Year with the biggest difference in population percentage: {biggest_diff_year}")

    plt.legend()
    plt.show()

    if should_display:
        print("Różnorodność wśród imion rosła do 1910 roku. Malała w latach 1910-1960 i zaczęła bardzo rosnąć od 1960 roku.")
        print("Różnorodność wśród imion zaley od płci.")

    return top_males, top_females, all_top_1000


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

def zad10(df: pd.DataFrame, names_num=1000):
    top_1000_males, top_1000_females, _ = zad6(df, names_num, should_display=False)
    data, now_female_name, now_male_name = get_changing_names(df, top_1000_females, top_1000_males)

    males = data[data["Gender"] == "M"]
    females = data[data["Gender"] == "F"]

    females.loc[:, "ratio"] = 1 - females["ratio"]

    comp = males.merge(females, on=["Year", "Name"], how='outer')
    comp["ratio"] = comp.apply(lambda row: row["ratio_y"] if pd.isna(row["ratio_x"]) else row["ratio_x"], axis=1)

    now_female_data = comp[comp["Name"] == now_female_name[0]]
    now_male_data = comp[comp["Name"] == now_male_name[0]]

    print(f"Kiedyś męskie, obecnie żeńskie imię: {now_female_name[0]}")
    print(f"Kiedyś żeńskie, obecnie męskie imię: {now_male_name[0]}")

    plt.figure(figsize=(10, 6))
    plt.plot(now_female_data["Year"], now_female_data["ratio"], marker='o', linestyle='-', label=f"Popularity of name {now_female_name[0]}")
    plt.plot(now_male_data["Year"], now_male_data["ratio"], marker='o', linestyle='-', label=f"Popularity of name {now_male_name[0]}")
    plt.axhline(y=0.5, color='gray', linestyle='--')
    plt.xlabel("Year")
    plt.ylabel("Ratio")
    plt.title(f"Change of names popularity across both genders")
    plt.grid(True)
    plt.legend()

    plt.text(now_female_data["Year"].min(), 0.9, "Imiona męskie", fontsize=12, color='blue')
    plt.text(now_female_data["Year"].min(), 0.1, "Imiona żeńskie", fontsize=12, color='red')

    plt.show()


def get_changing_names(df, top_1000_females, top_1000_males):
    # funkcja uzyta w zad10
    data = get_gender_popularity(df, top_1000_females, top_1000_males)
    # data["shifted_ratio"] = data["ratio"].shift(1)
    first_period = data.loc[data["Year"].between(1880, 1919)]
    first_period_females_grouped, first_period_males_grouped = prepare_gender_popularity_period(first_period)
    # print(pd.crosstab(index=[first_period_males["Name"], first_period_males["Year"]], columns=first_period_males["Year"], values=first_period_males["ratio"], aggfunc='mean'))
    second_period = data.loc[data["Year"].between(2001, 2023)]
    second_period_males = second_period[second_period["Gender"] == "M"]
    second_period_females = second_period[second_period["Gender"] == "F"]
    second_period_males_grouped = second_period_males.groupby(["Name"]).agg({"ratio": "mean"})
    second_period_females_grouped = second_period_females.groupby(["Name"]).agg({"ratio": "mean"})
    mr = first_period_males_grouped.merge(second_period_females_grouped, on=["Name"])
    mr2 = first_period_females_grouped.merge(second_period_males_grouped, on=["Name"])
    mr["fin"] = (mr["ratio_x"] + mr["ratio_y"]) / 2
    mr2["fin"] = (mr2["ratio_x"] + mr2["ratio_y"]) / 2
    now_female_name = mr["fin"].sort_values(ascending=False).head(1).reset_index()["Name"]
    now_male_name = mr2["fin"].sort_values(ascending=False).head(1).reset_index()["Name"]
    return data, now_female_name, now_male_name


def prepare_gender_popularity_period(first_period):
    # funkcja uzyta w zad 11 oraz w funkcji get_changing_names
    first_period_males = first_period[first_period["Gender"] == "M"]
    first_period_females = first_period[first_period["Gender"] == "F"]
    first_period_males_grouped = first_period_males.groupby(["Name"]).agg({"ratio": "mean"})
    first_period_females_grouped = first_period_females.groupby(["Name"]).agg({"ratio": "mean"})
    return first_period_females_grouped, first_period_males_grouped


def get_gender_popularity(df, top_1000_females, top_1000_males):
    # funkcja uzyta w zad 11 oraz w funkcji get_changing_names
    male_names = top_1000_males["Name"].tolist()
    female_names = top_1000_females["Name"].tolist()
    common_names = list(set(male_names) & set(female_names))
    data = df.loc[df["Name"].isin(common_names)]
    data_total_names = data.groupby(["Year", "Name"]).agg({"Count": "sum"}).rename(
        columns={"Count": "total_count"}).reset_index()
    data = data.merge(data_total_names, on=["Year", "Name"])
    data["ratio"] = data["Count"] / data["total_count"]
    return data


def zad11():
    conn = sqlite3.connect("./data/names_pl_2000-23.sqlite")  # połączenie do bazy danych - pliku
    df = pd.read_sql_query("SELECT * FROM males UNION ALL SELECT * FROM females", conn)
    df = df.rename(columns={"Imię": "Name", "Liczba": "Count", "Rok": "Year", "Płeć": "Gender"})
    df['Gender'] = df['Gender'].replace('K', 'F')
    conn.close()

    # Getting 200 most popular names:

    top_names, _, all_top = zad8(df, 200, False, should_display=False)

    names = list(set(top_names["Name"]))
    top_names = all_top.loc[all_top["Name"].isin(names)]

    years = [2000, 2013, 2023]

    filtered_data = top_names[top_names["Year"].isin(years)]

    # Plot the histogram
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    plt.xlabel("Popularity")
    plt.ylabel("Frequency")
    colors = ['blue', 'green', 'red']
    for i, year in enumerate(years):
        yearly_data = filtered_data[filtered_data["Year"] == year]
        axes[i].hist(yearly_data["popularity"], bins=20, alpha=0.5, label=str(year), color=colors[i])
        axes[i].legend()
        axes[i].set_ylim(0, 120)

    plt.title("Histogram of Name Popularity for Selected Years")

    plt.show()

    print("Możemy zauważyc podobne trendy w porównaniu z USA, lecz wzrost zróznicowania nie jest aż tak duży jak w USA.")
    print("Analizujac histogram mozemy zauważyc, że na przestrzeni lat zmniejszyła się ilość nadawania bardzo popularnych imion")
    print("Zmiana może wynikać z wstąpienia do Uni Europejskiej, napływu imigrantów, silnym wpływie kultury zachodniej i amerykańskiej")


    # Getting most gender neutral names:

    data = get_gender_popularity(df, df[df["Gender"] == "F"], df[df["Gender"] == "M"])
    first_period_females_grouped, first_period_males_grouped = prepare_gender_popularity_period(data)
    mr = first_period_males_grouped.merge(first_period_males_grouped, on=["Name"])
    mr["fin"] = (mr["ratio_x"] + mr["ratio_y"]) / 2
    mr = mr.sort_values(by="fin", ascending=False).head(2)
    print("Imiona często nadawane chłopcom i dziewczynkom:")
    print(mr.reset_index()["Name"])

def main():
    df = get_txt_dataset()
    zad2(df)
    zad3(df)
    zad4(df)
    zad5(df)
    zad6(df)
    zad7(df)
    zad8(df)
    zad9(df)
    zad10(df)

    zad11()

if __name__ == '__main__':
    main()
