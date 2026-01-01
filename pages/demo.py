import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.style.use('ggplot')
import seaborn as sns
import shap

st.set_page_config(layout="wide")

from utils import ratio_health_gdp, passthrough_featurenames
import __main__                 
__main__.ratio_health_gdp = ratio_health_gdp
__main__.passthrough_featurenames = passthrough_featurenames 

import joblib 
from load_utils import load_models, load_df, build_sidebar, shap_plot, shap_decision_plot

# ----------------------------------
# LOAD MODELS
#-----------------------------------
qr_models = load_models()

SHIFT = {
    "q25": -0.3824,
    "q50": -0.6396,
    "q75": -0.5392
} 
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
                \nAs of 2018, it is assigned to the :orange[*{base_df['world_income_group'].iloc[5].lower()}*] group.
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
        pred_low  = qr_models["low"].predict(X_original) + SHIFT["q25"],
        pred_med  = qr_models["med"].predict(X_original) + SHIFT["q50"],
        pred_high = qr_models["high"].predict(X_original) + SHIFT["q75"]
        #pred_low  = qr_models["low"].predict(X_original),
        #pred_med  = qr_models["med"].predict(X_original),
        #pred_high = qr_models["high"].predict(X_original)
    )
    #predicts_original
    # ----------------------------------
    # Q-MODELS PREDICTIONS WITH NEW DF
    #----------------------------------- 
    X_new = modified_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme", "pred_q025", "pred_q05", "pred_q075", "q05_pos", "q075_pos", "bandwidth", "bandwidth_pos"])
    predicts_new = modified_df.assign(
        pred_low  = qr_models["low"].predict(X_new) + SHIFT["q25"],
        pred_med  = qr_models["med"].predict(X_new) + SHIFT["q50"],
        pred_high = qr_models["high"].predict(X_new) + SHIFT["q75"]
    )
        
    st.divider()
    # ----------------------------------
    # SHOW REFERENCE Q-MODELS PREDICTIONS 
    # WITHOUT ANY SIMULATION
    #----------------------------------- 
    q05_pos = predicts_original["q05_pos"].median()
    bw_med = predicts_original["bandwidth"].median()
    bw_med_pos = predicts_original["bandwidth_pos"].median()
    
    st.markdown(f"##### :orange[{base_df['Entity'].iloc[0]}'s] reference predicted child mortality rate for {' | '.join(map(str, sorted(years_select)))}")
    st.write(f"Based on a global reference, the corresponding quantile prediction specific to {base_df['Entity'].iloc[0]} in comparison to the remaining data is highlighted.")
    st.write("For convenience, each quantile reference prediction shows the median U5MR value if multiple years are chosen. The charts below show all predicted chosen samples.")
    st.markdown(f"###### Without any indicator adjustments the U5MR predictions for {base_df['Entity'].iloc[0]} are:")
    
    focus_quant_025 = False
    focus_quant_05 = False
    focus_quant_075 = False
    if (q05_pos >= 0.75) or (bw_med_pos >= 0.75):
        focus_quant_075 = True
    elif (q05_pos <= 0.25) and (bw_med_pos <= 0.5):
        focus_quant_025 = True
    else:
        focus_quant_05 = True
    
    quant_base1, quant_base2, quant_base3 = st.columns(3, border=True)
    with quant_base1:
        st.metric(
            label="*Q 0.25 prediction*", 
            value=f"{predicts_original['pred_low'].median():.2f} per 1000",
            chart_data=predicts_original['pred_low'].tolist(),
            chart_type="line", 
        )
        st.write(f"In 75% of cases with comparable feature combinations, the true value is above {predicts_original['pred_low'].median():.2f} per 1000")
        if focus_quant_025:
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
        if focus_quant_05:
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
        if focus_quant_075:
            with st.container():
                st.error("Focus Quantile")
                
    if focus_quant_025 == True:
        st.info(f"""Why is the Q0.25 quantile the focus? For **{base_df['Entity'].iloc[0]}** the prediction uncertainty is relatively low.
                \nThe range of the prediction between 'bottom 0.25 quantile' and 'top 0.75 quantile' is only {bw_med:.2f}. Thus the outcome seems robust. 
                \nGlobally compared, **{base_df['Entity'].iloc[0]}** is among the {(bw_med_pos * 100):.2f}% with the lowest uncertainty.""")
    if focus_quant_05 == True:
        st.warning(f"""Why is the Q0.5 quantile the focus? For **{base_df['Entity'].iloc[0]}** the prediction uncertainty is moderate.
                \nThe range of the prediction between 'bottom 0.25 quantile' and 'top 0.75 quantile' is {bw_med:.2f}. 
                \nGlobally compared, around {(bw_med_pos * 100):.2f}% of the remaining country data samples are below this uncertainty range.""")
    if focus_quant_075 == True:
        st.error(f"""Why is the Q0.75 quantile the focus? For **{base_df['Entity'].iloc[0]}** the prediction uncertainty is relatively high.
                \nThe range of the prediction between 'bottom 0.25 quantile' and 'top 0.75 quantile' is {bw_med:.2f}. Thus the outcome seems more risky. 
                \nGlobally compared, **{base_df['Entity'].iloc[0]}** is among the {abs(-((bw_med_pos * 100) - (100-bw_med_pos))):.2f}% with the highest uncertainty.""")
    
    #st.write(df_ref.loc[df_ref["world_income_group"].isin(years_df["world_income_group"]), "bandwidth_pos"].mean())
    #df_ref.loc[df_ref["world_income_group"].isin(years_df["world_income_group"])]
    
    #st.write(f"Table of indicator values for **{base_df['Entity'].iloc[0]}**")
    X_original
    st.space()
    st.write(f"How the features affect the prediction for {X_original['world_regions_wb'].iloc[0]} countries including **{base_df['Entity'].iloc[0]}**")
    shap_by_region = df_ref.copy()
    if focus_quant_025:
        shap_plot(qr_models, shap_by_region, "low", "Focus Quantile 0.25")
        shap_decision_plot(qr_models, X_original, "low", "Focus Quantile 0.25")
    elif focus_quant_05:
        shap_plot(qr_models, shap_by_region, "med", "Focus Quantile 0.5")
        shap_decision_plot(qr_models, X_original, "med", "Focus Quantile 0.5")
    else:
        shap_plot(qr_models, shap_by_region, "high", "Focus Quantile 0.75")
        shap_decision_plot(qr_models, X_original, "high", "Focus Quantile 0.75")
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
            label="*Q 0.25 prediction* (Best Case)", 
            delta_color="inverse",
            value=f"{predicts_new['pred_low'].median():.2f} per 1000", 
            delta=f"""{predicts_new['pred_low'].median() - predicts_original['pred_low'].median():.2f} per 1000
                        ({ ((predicts_new['pred_low'].median() - predicts_original['pred_low'].median()) / predicts_original['pred_low'].median() * 100):.2f} %)""",
            chart_data=predicts_new['pred_low'].tolist(),
            chart_type="line",
        )
        if focus_quant_025:
            with st.container():
                st.success("Focus Quantile")
    with quant_col2:
        st.metric(
            label="*Q 0.5 (median) prediction*", 
            delta_color="inverse",
            value=f"{predicts_new['pred_med'].median():.2f} per 1000", 
            delta=f"""{predicts_new['pred_med'].median() - predicts_original['pred_med'].median():.2f} per 1000
                    ({ ((predicts_new['pred_med'].median() - predicts_original['pred_med'].median()) / predicts_original['pred_med'].median() * 100):.2f} %)""",
            chart_data=predicts_new['pred_med'].tolist(),
            chart_type="line",
        )
        if focus_quant_05:
            with st.container():
                st.success("Focus Quantile")  
    with quant_col3:
        st.metric(
            label="*Q 0.75 prediction* (Worst Case)", 
            delta_color="inverse",
            value=f"{predicts_new['pred_high'].median():.2f} per 1000", 
            delta=f"""{predicts_new['pred_high'].median() - predicts_original['pred_high'].median():.2f} per 1000
                    ({ ((predicts_new['pred_high'].median() - predicts_original['pred_high'].median()) / predicts_original['pred_high'].median() * 100):.2f} %)""",
            chart_data=predicts_new['pred_high'].tolist(),
            chart_type="line",
        )
        if focus_quant_075:
            with st.container():
                st.success("Focus Quantile")
                
    if(len(years_df) > 1):
        st.space() 
        st.markdown(f"###### How the adjusted features affect the new prediction") 
        if focus_quant_025:
            shap_decision_plot(qr_models, X_new, "low", "Focus Quantile 0.25")
        elif focus_quant_05:
            shap_decision_plot(qr_models, X_new, "med", "Focus Quantile 0.5")
        else:
            shap_decision_plot(qr_models, X_new, "high", "Focus Quantile 0.75")
            
            
    st.space()    
    df_preds = pd.DataFrame({
    "base_q025": predicts_original["pred_low"],
    "whatif_q025": predicts_new["pred_low"],
    "base_q05": predicts_original["pred_med"],
    "whatif_q05": predicts_new["pred_med"],
    "base_q075": predicts_original["pred_high"],
    "whatif_q075": predicts_new["pred_high"],
    "base_year": predicts_original["Year"],
    "whatif_year": predicts_new["Year"]
    })    
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        if len(predicts_original["Year"]) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(data=df_preds, x="base_year", y="base_q05", label="base median prediction", linewidth=3)
            sns.lineplot(data=df_preds, x="whatif_year", y="whatif_q05", label="simulated median prediction", linewidth=3)
            ax.fill_between(df_preds["base_year"], df_preds["base_q025"], df_preds["base_q075"], alpha=0.25)
            ax.fill_between(df_preds["whatif_year"], df_preds["whatif_q025"], df_preds["whatif_q075"], alpha=0.25)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("U5MR per 1000 live births")
            ax.set_xlabel("Year")
            ax.set_title(f"Reference vs. Simulated U5MR Prediction For {base_df['Entity'].iloc[0]}")
            st.pyplot(fig)
    
    st.divider()    
    st.info(f"ORIGINAL PRED (25%, 50%, 75%): "
        f"{predicts_original['pred_low'].median():.2f}, {predicts_original['pred_med'].median():.2f}, {predicts_original['pred_high'].median():.2f}")
    st.info(f"NEW PRED (25%, 50%, 75%): "
        f"{predicts_new['pred_low'].tolist()}, {predicts_new['pred_med'].median():.2f}, {predicts_new['pred_high'].median():.2f}")

    st.divider()      
    expander = st.expander(f"For reference, view actual historical child mortality rates for {base_df['Entity'].iloc[0]} (ground truth)")
    for i, (idx, row) in enumerate(years_df.iterrows()):        
        expander.write(f'{row["Year"]}: {row["child_mortality_igme"]:.2f} per 1000')