import os

import pandas as pd


def read_df(rceth_csv_path):
    df = pd.read_csv(rceth_csv_path, encoding='windows-1251')
    df["full_name"] = df["trade_name"] + " " + \
        df["dosage_form"] + " " + df["manufacturer"]

    return df


def search_name(df, name):
    # Find name in dataframe and create subdataframe
    contain_names = df['trade_name'].str.contains(
        name.lower(), case=False)
    med_series = df.loc[(contain_names) & (
        df["dosage_form"] != "субстанция"), "full_name"]
    med_series = med_series.sort_values()

    # Sort values and convert to list
    names = med_series.to_list()

    return names


def get_instr_url(instr_urls):
    # Split string and strip each instruction
    instr_urls = instr_urls.split(",")
    instr_urls = [instr.strip() for instr in instr_urls]

    # Sort instruction by last letter in basename
    instr_urls.sort(key=lambda x: os.path.splitext(
        os.path.basename(x))[0][-1])

    return instr_urls[-1]
