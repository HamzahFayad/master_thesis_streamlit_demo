#----------------------------------------------------------------------
# data_preprocessing.py
#
# Define entire preprocessing ML pipeline
#----------------------------------------------------------------------


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PowerTransformer, OneHotEncoder, FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
import config


# ----------------------------------
# create ratio variable from healthspending & gdp
# @return FunctionTransformer
#-----------------------------------
def passthrough_featurenames(transformer, input_features=None):
    return ["gdp_per_capita_worldbank", 
            "annual_healthcare_expenditure_per_capita", 
            "healthspending_gdp_ratio"]

def ratio_health_gdp(X):
    X = X.copy()
    X["healthspending_gdp_ratio"] = (
        X["annual_healthcare_expenditure_per_capita"] - 
        X["gdp_per_capita_worldbank"]
    )
    return X

# ----------------------------------
# create interaction terms based on domain knowledge:
# many feature synergies  
# @return FunctionTransformer
#-----------------------------------
def create_interaction(df):
    df = df.copy()
    cat_cols = [c for c in df.columns if c.startswith('world_income_group_')]
    for col in cat_cols:
        df[f'health_gdp_ratio_x_{col}'] = df[col] * df['healthspending_gdp_ratio']
        df[f'nurses_midwives_per1000_x_{col}'] = df[col] * df['nurses_and_midwives_per_1000_people']
        df[f'physicians_per1000_x_{col}'] = df[col] * df['physicians_per_1000_people']
        df[f'undernourishment_x_{col}'] = df[col] * df['prevalence_of_undernourishment']
        df[f'share_popul_urban_x_{col}'] = df[col] * df['share_of_population_urban']
        df[f'share_without_water_x_{col}'] = df[col] * df['share_without_improved_water']
        df[f'vacc_coverage_x_{col}'] = df[col] * df['vaccination_coverage_who_unicef']
        df[f'schooling_x_{col}'] = df[col] * df['years_of_schooling']
    return df

new_interaction_terms = ["health_gdp_ratio_x_", "nurses_midwives_per1000_x_", "physicians_per1000_x_", "undernourishment_x_", 
                    "share_popul_urban_x_", "share_without_water_x_", "vacc_coverage_x_", "schooling_x_"]
def get_inter_names(transformer, input_features):
    cat_cols = [c for c in input_features if c.startswith('world_income_group_')]
    new_cols = [f'{i}{c}' for c in cat_cols for i in new_interaction_terms]
    return list(input_features) + new_cols

"""def passthrough_feature_int_names(transformer, input_features=None):
    return ["nurses_and_midwives_per_1000_people", "physicians_per_1000_people", 
            "prevalence_of_undernourishment","share_of_population_urban", "share_without_improved_water", 
            "vaccination_coverage_who_unicef","years_of_schooling", 
            "urban_x_medical_access", "urban_x_water_access"]
    
def feature_interactions(X):
    X = X.copy()
    total_medical_staff = X["nurses_and_midwives_per_1000_people"] + X["physicians_per_1000_people"]
    X["urban_x_medical_access"] = total_medical_staff * X["share_of_population_urban"]
    X["urban_x_water_access"] = X["share_of_population_urban"] * X["share_without_improved_water"]
    return X
"""
    
# ----------------------------------
# Preprocessing Pipeline Steps
#-----------------------------------
def preprocessing_pipeline():
    
    print(f"Initialize Preprocessing: Imputing, Transforming, Scaling, OHE...")

    # HEALTH/GDP RATIO FUNCTRANSFORMER
    ratio_feature = FunctionTransformer(
        func = ratio_health_gdp, 
        validate = False, 
        feature_names_out=passthrough_featurenames
        )
    
     # INTERACTION TERMS FUNCTRANSFORMER
    """interaction_terms = FunctionTransformer(
        func = feature_interactions, 
        validate = False, 
        feature_names_out=passthrough_feature_int_names
        )"""

    # IMPUTE AND TRANSFORM NUMERIC VARIABLES
    impute_transform = ColumnTransformer([
        
        ("pre_rightskewed", Pipeline([
            ("knn_impute1", KNNImputer(n_neighbors=5, weights="distance")),
            ("log_transform", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
        ]), config.right_skewed_cols),
        
        ("pre_leftskewed", Pipeline([
            ("knn_impute2", KNNImputer(n_neighbors=5, weights="distance")),
            ("power_transform", PowerTransformer(method="yeo-johnson"))
        ]), config.left_skewed_cols),
        
        ("pre_normal", Pipeline([
            ("knn_impute3", KNNImputer(n_neighbors=5, weights="distance")),
        ]), config.normal_cols),
            
    ], verbose_feature_names_out=False, remainder='passthrough').set_output(transform="pandas")

    # CREATE A RATIO COLUMN  
    ratio_he_gdp = ColumnTransformer([
        
        ("health_gdp_ratio", Pipeline([
            ("health_gdp_ratio", ratio_feature),
        ]), [config.col_gdp, config.col_healthspending])
        
    ], verbose_feature_names_out=False, remainder="passthrough").set_output(transform="pandas")
    
    # CREATE INTERACTION COLUMNS  
    """f_interactions = ColumnTransformer([
    
        ("int_terms", Pipeline([
            ("interactions", interaction_terms),
        ]), config.rest_nums)
    
    ], verbose_feature_names_out=False, remainder="passthrough").set_output(transform="pandas")
    """

    # ONE HOT ENCODE CATEGORIC VARIABLES
    ohe_cats = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # SCALE NUM VARIABLES & USE OHE ON CAT VARIABLES
    scale_ohe_step = ColumnTransformer([
        
        ("drop_num_cols", "drop", [config.col_gdp, config.col_healthspending]),
        ("scale_nums", RobustScaler(), config.rest_nums + ["healthspending_gdp_ratio"]), #"urban_x_medical_access", "urban_x_water_access"]),
        ("ohe_cats", ohe_cats, [config.col_regions, config.col_incomegroup]),
        
    ], verbose_feature_names_out=False, remainder="passthrough").set_output(transform="pandas")

    # END PIPELINE TO COMBINE ALL PREPROCESSING STEPS
    end_pipe = Pipeline([
        
        ("prep_nums", impute_transform),
        ("ratio_feature", ratio_he_gdp),
        #("interaction_terms", f_interactions),
        ("scale_ohe", scale_ohe_step),
        ('interaction', FunctionTransformer(create_interaction, feature_names_out=get_inter_names)),
        #("final_impute", KNNImputer(n_neighbors=5, weights="distance"))
        
    ]).set_output(transform="pandas")
    
    print(f"Preprocessing finished...")
    
    return end_pipe