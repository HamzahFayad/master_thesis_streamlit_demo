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

import __main__
__main__.ratio_health_gdp = ratio_health_gdp
__main__.passthrough_featurenames = passthrough_featurenames