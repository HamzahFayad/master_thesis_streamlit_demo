import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import joblib

from load_utils import load_models, load_df, build_sidebar

from utils import ratio_health_gdp

# ----------------------------------
# LOAD MODELS
#-----------------------------------
qr_models = load_models()

# ----------------------------------
# LOAD REFERENCE DATAFRAME
#-----------------------------------
df_ref = load_df()

# ----------------------------------
# RESET SIMULATE BUTTON STATE
#-----------------------------------
def reset_button():
    st.session_state.simulate_btn = False

if "simulate_btn" not in st.session_state:
    st.session_state.simulate_btn = False
    
# ----------------------------------
# CHOOSE COUNTRY & YEARS
#-----------------------------------
country_select = st.sidebar.selectbox(
    label="Select a country",
    options=df_ref["Entity"].unique().tolist(),
    index=None,
    placeholder="Country",
    on_change=reset_button
)
base_df = df_ref[df_ref["Entity"] == country_select].copy()

years_select = st.sidebar.multiselect(
    label="Select reference year(s)",
    options=base_df["Year"].tolist(),
    default=[],
    placeholder="Available Years",
    on_change=reset_button
)
years_df = base_df[base_df["Year"].isin(years_select)].copy()

# ----------------------------------
# SIMULATOR INTRO
#-----------------------------------
if country_select is not None:
    st.title(f"Simulate :orange[{base_df['Entity'].iloc[0]}'s] U5MR")
    st.markdown(f"""
                {base_df['Entity'].iloc[0]}'s region as defined by the World Bank: :orange[*{base_df['world_regions_wb'].iloc[0]}*]
                \nTo find out how the under-five mortality rate outcome would have differed under different conditions of socioeconomic and health-related factors,
                select one or multiple years and adjust given indicators to simulate the under-five mortality rate for {base_df["Entity"].iloc[0]}.""")  
else:
    st.title("Simulator")
    st.markdown("#### :orange['What-if'] prediction of under-five mortality rate at country level")
    st.markdown("Hypothetical scenarios")  
    
# ----------------------------------
# NEW CHANGES DF TO PREDICT WITH
#-----------------------------------
modified_df = years_df.copy()

# ----------------------------------
# SIDEBAR TO ADJUST INDICATORS
#-----------------------------------
indicators = build_sidebar(years_df, years_select, country_select)


if years_select and country_select is not None:
    
    # ----------------------------------
    # NEW CHANGED DF FOR PREDICTION
    #-----------------------------------
    modified_df = years_df.assign(
        annual_healthcare_expenditure_per_capita = lambda x: x["annual_healthcare_expenditure_per_capita"] * (1 + indicators["ahec"] / 100),
        gdp_per_capita_worldbank = lambda x: x["gdp_per_capita_worldbank"] * (1 + indicators["gdp"] / 100),
        nurses_and_midwives_per_1000_people = lambda x: x["nurses_and_midwives_per_1000_people"] * (1 + indicators["nm"] / 100),
        physicians_per_1000_people = lambda x: x["physicians_per_1000_people"] * (1 + indicators["phys"] / 100),
        prevalence_of_undernourishment = lambda x: x["prevalence_of_undernourishment"] + indicators["undernourishment"],
        share_of_population_urban = lambda x: x["share_of_population_urban"] + indicators["urban"],
        share_without_improved_water = lambda x: x["share_without_improved_water"] + indicators["water"],
        vaccination_coverage_who_unicef = lambda x: x["vaccination_coverage_who_unicef"] + indicators["vaccination"],
        years_of_schooling = lambda x: x['years_of_schooling'] + indicators["school"]
    )  
    
    # ----------------------------------
    # Q-MODELS PREDICTIONS WITH ORIGINAL DF
    #----------------------------------- 
    X_original = years_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme", "pred_q025", "pred_q05", "pred_q075", "q05_pos", "q075_pos", "bandwidth", "bandwidth_pos"])
    predicts_original = years_df.assign(
        pred_low  = qr_models["low"].predict(X_original),
        pred_med  = qr_models["med"].predict(X_original),
        pred_high = qr_models["high"].predict(X_original)
    )
    
    #predicts_original
    # ----------------------------------
    # Q-MODELS PREDICTIONS WITH NEW DF
    #----------------------------------- 
    X_new = modified_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme", "pred_q025", "pred_q05", "pred_q075", "q05_pos", "q075_pos", "bandwidth", "bandwidth_pos"])
    predicts_new = modified_df.assign(
        pred_low  = qr_models["low"].predict(X_new),
        pred_med  = qr_models["med"].predict(X_new),
        pred_high = qr_models["high"].predict(X_new)
    )
        
    st.divider()
    # ----------------------------------
    # SHOW REFERENCE Q-MODELS PREDICTIONS 
    # WITHOUT ANY SIMULATION
    #----------------------------------- 
    q05_pos = predicts_original["q05_pos"].median()
    #q075_pos = predicts_original["q075_pos"].median()
    bw_med = predicts_original["bandwidth"].median()
    bw_med_pos = predicts_original["bandwidth_pos"].median()
    
    st.markdown(f"##### :orange[{base_df['Entity'].iloc[0]}'s] reference predicted child mortality rate for {' | '.join(map(str, sorted(years_select)))}")
    st.write(f"Based on a global reference, the corresponding quantile prediction specific to {base_df['Entity'].iloc[0]} in comparison to the remaining data is highlighted.")
    st.write("For convenience, each quantile reference prediction shows the median U5MR value if multiple years are chosen. The charts below show all predicted chosen samples.")
    st.markdown(f"###### Without any indicator adjustments the U5MR predictions for {base_df['Entity'].iloc[0]} are:")
    
    highlight_col1 = False
    highlight_col2 = False
    highlight_col3 = False
    if (q05_pos >= 0.75) or (bw_med_pos >= 0.75):
        highlight_col3 = True
    elif (q05_pos <= 0.25) and (bw_med_pos <= 0.5):
        highlight_col1 = True
    else:
        highlight_col2 = True
    
    quant_base1, quant_base2, quant_base3 = st.columns(3, border=True)
    with quant_base1:
        st.metric(
            label="*Q 0.25 prediction*", 
            value=f"{predicts_original['pred_low'].median():.2f} per 1000",
            chart_data=predicts_original['pred_low'].tolist(),
            chart_type="line", 
        )
        st.write(f"In 75% of cases with comparable feature combinations, the true value is above {predicts_original['pred_low'].median():.2f} per 1000")
        if highlight_col1:
            with st.container():
                st.info("Focus Quantile")
    with quant_base2:
        st.metric(
            label="*Q 0.5 (median) prediction*", 
            value=f"{predicts_original['pred_med'].median():.2f} per 1000", 
            chart_data=predicts_original['pred_med'].tolist(),
            chart_type="line", 
        )
        st.write(f"In 50% of cases with comparable feature combinations, the true value is between {predicts_original['pred_low'].median():.2f} and {predicts_original['pred_high'].median():.2f} per 1000")   
        if highlight_col2:
            with st.container():
                st.warning("Focus Quantile") 
    with quant_base3:
        st.metric(
            label="*Q 0.75 prediction*", 
            value=f"{predicts_original['pred_high'].median():.2f} per 1000", 
            chart_data=predicts_original['pred_high'].tolist(),
            chart_type="line", 
        )
        st.write(f"In 75% of cases with comparable feature combinations, the true value is below {predicts_original['pred_high'].median():.2f} per 1000")
        if highlight_col3:
            with st.container():
                st.error("Focus Quantile")
                
    if highlight_col1 == True:
        st.info(f"""Why is the Q0.25 quantile the focus? For **{base_df['Entity'].iloc[0]}** the prediction uncertainty is relatively low.
                \nThe estimated range of the prediction between 'bottom 0.25 quantile' and 'top 0.75 quantile' is only {bw_med:.2f}. 
                \nGlobally compared, **{base_df['Entity'].iloc[0]}** is among the {(bw_med_pos * 100):.2f}% with the lowest uncertainty.""")
    if highlight_col2 == True:
        st.warning(f"""Why is the Q0.5 quantile the focus? For **{base_df['Entity'].iloc[0]}** the prediction uncertainty is moderate.
                \nThe estimated range of the prediction between 'bottom 0.25 quantile' and 'top 0.75 quantile' is {bw_med:.2f}. 
                \nGlobally compared, around {(bw_med_pos * 100):.2f}% of the remaining country data samples are below this uncertainty range.""")
    if highlight_col3 == True:
        st.error(f"""Why is the Q0.75 quantile the focus? For **{base_df['Entity'].iloc[0]}** the prediction uncertainty is relatively high.
                \nThe estimated range of the prediction between 'bottom 0.25 quantile' and 'top 0.75 quantile' is {bw_med:.2f}. 
                \nGlobally compared, **{base_df['Entity'].iloc[0]}** is among the {abs(-((bw_med_pos * 100) - (100-bw_med_pos))):.2f}% with the highest uncertainty.""")
    #years_df 
    #predicts_original
    st.divider()
    
        
        
        
        
    #u5mr_mean = years_df['child_mortality_igme'].mean()
    #u5mr_median = years_df['child_mortality_igme'].median()
    #u5mr_min = years_df['child_mortality_igme'].min()
    #u5mr_max = years_df['child_mortality_igme'].max()
    #st.write(f":orange[{base_df['Entity'].iloc[0]}'s] actual observed child mortality rate for {' | '.join(map(str, sorted(years_select)))}")
    #st.write(f"Mean: {u5mr_mean:.2f}")
    #st.write(f"Median: {u5mr_median:.2f}")
    #if len(years_select) > 1:
    #    st.write(f"Ranges from {u5mr_min:.2f} - {u5mr_max:.2f}")
        
    #st.write("Modified")
    #modified_df
    
    
    
    
    

if st.session_state.simulate_btn and (years_select and country_select is not None):
    # ----------------------------------
    # GET BEST FIT QUANTILE MODEL (BASED ON REAL MEAN TARGET)
    #----------------------------------- 
    real_u5mr_median = years_df['child_mortality_igme'].median()
    #st.write(f"Real Mean U5MR: {real_u5mr_mean}")
    quant_errors = {
    "Q0.25": abs(real_u5mr_median - predicts_original['pred_low'].median()),
    "Q0.5": abs(real_u5mr_median - predicts_original['pred_med'].median()),
    "Q0.75": abs(real_u5mr_median - predicts_original['pred_high'].median())
    }
    best_q = min(quant_errors, key=quant_errors.get)
    delta_res = quant_errors[best_q]
    #----------------------------------- 

    # ----------------------------------
    # SHOW Q-MODELS PREDICTIONS 
    # AFTER SIMULATION
    #----------------------------------- 
    st.markdown(f"##### After indicator adjustments, :orange[{base_df['Entity'].iloc[0]}'s] simulated prediction of U5MR for {' | '.join(map(str, sorted(years_select)))}")

    quant_col1, quant_col2, quant_col3 = st.columns(3, border=True)  
    with quant_col1:
        st.metric(
            label="*Q 0.25 prediction*", 
            delta_color="inverse",
            value=f"{predicts_new['pred_low'].median():.2f} per 1000", 
            delta=f"{predicts_new['pred_low'].median() - predicts_original['pred_low'].median():.2f} per 1000",
            chart_data=predicts_new['pred_low'].tolist(),
            chart_type="line",
        )
    with quant_col2:
        st.metric(
            label="*Q 0.5 (median) prediction*", 
            delta_color="inverse",
            value=f"{predicts_new['pred_med'].median():.2f} per 1000", 
            delta=f"{predicts_new['pred_med'].median() - predicts_original['pred_med'].median():.2f} per 1000",
            chart_data=predicts_new['pred_med'].tolist(),
            chart_type="line",
        )    
    with quant_col3:
        st.metric(
            label="*Q 0.75 prediction*", 
            delta_color="inverse",
            value=f"{predicts_new['pred_high'].median():.2f} per 1000", 
            delta=f"{predicts_new['pred_high'].median() - predicts_original['pred_high'].median():.2f} per 1000",
            chart_data=predicts_new['pred_high'].tolist(),
            chart_type="line",
        )
            
            
    st.divider()      
    st.info(f"ORIGINAL PRED (25%, 50%, 75%): "
        f"{predicts_original['pred_low'].median():.2f}, {predicts_original['pred_med'].median():.2f}, {predicts_original['pred_high'].median():.2f}")
    st.info(f"NEW PRED (25%, 50%, 75%): "
        f"{predicts_new['pred_low'].tolist()}, {predicts_new['pred_med'].median():.2f}, {predicts_new['pred_high'].median():.2f}")

    st.divider()      
    expander = st.expander(f"For reference, view actual historical child mortality rates for {base_df['Entity'].iloc[0]}")
    for i, (idx, row) in enumerate(years_df.iterrows()):        
        expander.write(f'{row["Year"]}: {row["child_mortality_igme"]:.2f} per 1000')