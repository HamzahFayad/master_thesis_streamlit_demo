# Config File for main df & variables
#----------------------------------------------------------------------
# config.py
#
# create prepared dataset, 
# fefine all variables
#----------------------------------------------------------------------

import pandas as pd
from data_cleaning import load_merge_raw_data, exclude_countries_high_missing_values

#Data Path
DATA_PATH = "./00_data/0_raw/"

"""
PREPARE DATASETS
@return main prepared dataframe to 
"""
def data_preparation():
    #STEP 1: Load & Merge DATASETS, EXCLUDE NON-COUNTRIES & LIMIT TO 6-YEAR PERIOD PER COUNTRY
    merged_df = load_merge_raw_data(DATA_PATH)
    #STEP 2: REMOVE COUNTRIES WITH HIGH AMOUNT OF MISSING VALUES (>=50%)
    df = exclude_countries_high_missing_values(merged_df)
    #print("Prepared Main Dataset:\n", df)
    print(f"Start ML process for prepared df with {len(df)} rows...")
    return df

prepared_df = data_preparation()

#low_incomes = ['High-income countries', 'Upper-middle-income countries']
#prepared_df = prepared_df[prepared_df['world_income_group'].isin(low_incomes)]
# ----------------------------------
# VARIABLES: Features, Target, Group
#-----------------------------------

y = prepared_df["child_mortality_igme"]
X = prepared_df.drop(columns=["Code", "Entity", "Year", "child_mortality_igme"])
group = prepared_df["Entity"]

num_variables = X.drop(columns=["world_regions_wb", "world_income_group"]).columns.to_list()
cat_variables = ["world_regions_wb", "world_income_group"]

normal_cols = ["years_of_schooling", "share_of_population_urban"]
left_skewed_cols = ["vaccination_coverage_who_unicef"]
norm_left = normal_cols + left_skewed_cols

right_skewed_cols = num_variables.copy() 
for el in norm_left:
    if el in right_skewed_cols:
        right_skewed_cols.remove(el)
        
#all numeric variables
all_numeric_cols = right_skewed_cols + left_skewed_cols + normal_cols

#categoric variables
col_country = "Entity"             
col_regions = "world_regions_wb"  
col_incomegroup = "world_income_group"

#cols to combine
col_healthspending = "annual_healthcare_expenditure_per_capita"
col_gdp = "gdp_per_capita_worldbank"

rest_nums = num_variables[2:]
#print(X.columns)