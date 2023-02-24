import os
import pandas as pd
import numpy as np
import re

# read original csv
data_path = os.path.join("data", "imgt_I_domain_1_2.csv")
df = pd.read_csv(data_path, index_col=0)

# extract protein info
protein_type = df["allele_full_name"].apply(lambda x: re.split('\W+', x)[1:4])
df[['code0', 'code1', 'code2']] = pd.DataFrame(list(protein_type), columns=['code0', 'code1', 'code2'])
df = df[['code0', 'code1', 'code2', 'domain_alpha1_2_sequence']]

# transform protein types to int number
df["code0"].replace({"A":0, "B":1, "C":2}, inplace=True)
df[['code1', 'code2']] = df[['code1', 'code2']].astype(int)

# transform protein types to consecutive int number, for one-hot encoding in training
def reindex_column(df, colname):
    my_map = pd.DataFrame(df[colname].unique()).to_dict()[0]
    inv_map = {v: k for k, v in my_map.items()}
    df[colname].replace(inv_map, inplace=True)
    
reindex_column(df, "code1") #code1: 0-63
reindex_column(df, "code2") #code2: 0-1026

# combine the rare cases into one type
def combine_minority(df, colname, threshold):
    target = len(df[colname].unique()) # code1:64, code2:1027
    # find out the code type, whose occurence is under threshold
    if_under_thd = df[colname].value_counts().sort_values() < threshold 
    comb_idx = list(if_under_thd[if_under_thd].index)
    # replace the minorities by the target type
    comb_dict = {i:target for i in comb_idx}  
    df[colname].replace(comb_dict, inplace=True)

combine_minority(df, "code1", 50)
combine_minority(df, "code2", 20)

df.to_csv(os.path.join("data", "protein_data.csv"))
