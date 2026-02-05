#----------------------------------------------------------------------
#data_cleaning.py
#
# Data integration: Load raw datasets (csv) from 00_data/0_raw/ and merge 
#                   all datasets with child_mortality_igme (Label) as base
# Exlude non-countries (remove additional continents etc. from Our World in Data)
# Filter dataset by 6 year period (decision made through strategic analysis)
# Scale up Label to 1000 (common for epidemiological research of U5MR)
#
# First Data Pre-Cleaning: 
# Remove countries rows with high amount of missing values with threshold set at >= 50%
#----------------------------------------------------------------------

# Step A - load raw Data + merge all data with u5mr as base + filter 6 year period + scale up label to 1000

#Imports
import pandas as pd
import os
from datetime import date

#get all data files & sort
PATH = "./00_data/0_raw/"
all_files = [f for f in os.listdir(PATH)]
sorted_files = sorted(os.listdir(PATH))
#print(sorted_files)

#u5mr file at first position (base for merging)
label_file = sorted_files.pop(1)
sorted_files.insert(0, label_file)

#list of non-countries to exclude in 'Entity' column
EXCLUDE_NO_COUNTRIES = ["Africa", "Asia", "Europe", "European Union (27)", "High-income countries", "Low-income countries", "Lower-middle-income countries", 
                      "North America", "Oceania", "South America", "Upper-middle-income countries", "World"]

#create column names from files
def new_col_names(name):
    return os.path.basename(name).split('.')[0].replace('-', '_')

"""
Limit DF to 6 year period:
Loop in 6 year periods and append to list as tuples
extract tuple of minimum null count, 6 year period with least NaN values
@return filtered df
"""
def get_years_period(df):
    nulls_list = []
    null_count = 0
    year_idx = df.index.get_level_values(2)
    
    for begin in range(2000, year_idx.max() - 4):
        end = begin + 5
        df_six_years = df[(year_idx >= begin) & (year_idx <= end)]
        null_count = df_six_years.isna().sum().sum()
        nulls_list.append((begin, end, null_count)) #list of tuple (begin, end, null_values count) 
        #print(f"From {begin} - {end}, NaN values count: {null_count}")
        
    # get the minimum NaN value
    found_period = min(nulls_list, key=lambda n: n[2])
    df = df[(df.index.get_level_values(2) >= found_period[0]) & (df.index.get_level_values(2) <= found_period[1])]
    return df



# def limit_period(df) -> pd.DataFrame:
#     df = df[(df.index.get_level_values(2) >= 2013) & (df.index.get_level_values(2) <= 2018)]
#     return df

"""
Load and Merge all CSV files:
load raw data, join columns, 
exclude non-countries, 
set Multi-Index and merge all 10 df
scale U5MR up to 1000
@return merged, limited df
"""
def load_merge_raw_data(PATH) -> pd.DataFrame:
    
    #Additional helper csv (assign each country a region based on World Bank)
    world_regions = pd.read_csv("./00_data/1_interim/world-regions-worldbank.csv")
    world_regions = world_regions.drop(["Entity", "Year"], axis=1)
    world_regions = world_regions.rename(columns={"World regions according to WB": "world_regions_wb"})
    
    world_income_class = pd.read_csv("./00_data/1_interim/world-bank-income-groups.csv")
    world_income_class = world_income_class.drop(["Entity"], axis=1)
    world_income_class = world_income_class.rename(columns={"World Bank's income classification": "world_income_group"})
    
    
    big_df = None
    joins = ['Entity', 'Code', 'Year']
    
    print("Load and merge raw datasets, reduce dataset to limited year-periods per country...")
    
    for name in sorted_files:
        cols_names = new_col_names(name)
        
        df = pd.read_csv(os.path.join(PATH, name), usecols=[0, 1, 2, 3])
        df.columns = joins + [cols_names]
        
        df = df[~df['Entity'].isin(EXCLUDE_NO_COUNTRIES)]
    
        df = df.set_index(joins) 

        if big_df is None:
            big_df = df.copy() 
        else: 
            big_df = big_df.merge(
                df, 
                left_index=True, 
                right_index=True, 
                how='left' 
            )
    # get six years period with least NaNs and limit big_df
    big_df = get_years_period(big_df)   #big_df = limit_period(big_df)
    # scale u5mr to 1000 (common in research by UN IGME etc.)
    big_df["child_mortality_igme"] = big_df["child_mortality_igme"] * 10
    
    big_df = big_df.reset_index()       #big_df = big_df.reset_index(level=0)
    
    big_df = pd.merge(big_df, world_regions, on="Code", how="left")
    big_df = pd.merge(big_df, world_income_class, on=["Code", "Year"], how="left")

    #print(big_df)  
    return big_df


#load_merge = load_merge_raw_data(PATH)



"""
Exclude Countries from DF with Missing Values Threshold >= 50%
"""
THRESHOLD = 50
def exclude_countries_high_missing_values(merged_df) -> pd.DataFrame:
    
    all_missing_values = merged_df.isnull().groupby(merged_df["Entity"]).sum()
    # get sum of values per country for 9 main potential features: 
    values_count_per_country = merged_df.groupby(merged_df["Entity"]).size().iloc[0] * 9
    
    all_missing_values["total_missing"] = all_missing_values.sum(axis=1)    #total missing values
    all_missing_values["total_missing_%"] = round((all_missing_values["total_missing"] / values_count_per_country) * 100, 2)  #total missing values %

    top_missing_countries = all_missing_values.sort_values(ascending=False, by="total_missing_%")
    exclude_countries = top_missing_countries[top_missing_countries["total_missing_%"] >= THRESHOLD]

    filtered_df_01 = merged_df[~merged_df["Entity"].isin(exclude_countries.index.tolist())].copy()

    #print("NEW FILTERED DF", filtered_df_01)
    print("Exclude countries with high missing values rate (> 50%)...")
    return filtered_df_01