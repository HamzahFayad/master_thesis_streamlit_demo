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
from load_utils import load_models, load_df, build_sidebar, shap_plot, shap_decision_plot, force_plot, dependance_plot



# shifts to correct "Coverage"
SHIFT = {
    "q25": -0.72, #-0.3824, #-0.8974
    "q50": -1.29, #-0.6396, #-0.3152
    "q75": -0.31 #-0.5392  #-0.8829
} 

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
    st.cache_resource.clear()
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
                select one or multiple years and adjust given indicators to simulate the under-five mortality rate for {base_df["Entity"].iloc[0]}.
                \nKeep in mind that for each country, only a period of six years was used to train the ML model due to the importance of a high data quality.""")  
else:
    st.title("Simulator")
    st.markdown("#### :orange['What-if'] prediction of under-five mortality rate at country level")
    st.markdown("*Hypothetical scenarios:* Understand how the indicators affect the U5MR predictions")  
    
# ----------------------------------
# NEW CHANGES DF TO PREDICT WITH
#-----------------------------------
modified_df = years_df.copy()

#NEW_TEST
orig_global_df = df_ref.copy()
modified_global_df = df_ref.copy()

# ----------------------------------
# SIDEBAR TO ADJUST INDICATORS
#-----------------------------------
with st.sidebar:
    indicators = build_sidebar(years_df, years_select, country_select)


if years_select and country_select is not None:
    # ----------------------------------
    # NEW CHANGED DF FOR PREDICTION
    #-----------------------------------
    modified_df = years_df.assign(
        annual_healthcare_expenditure_per_capita = lambda x: x["annual_healthcare_expenditure_per_capita"] * (1 + indicators["ahec"] / 100),
        gdp_per_capita_worldbank = lambda x: x["gdp_per_capita_worldbank"] * (1 + indicators["gdp"] / 100),
        
        nurses_and_midwives_per_1000_people = lambda x: x["nurses_and_midwives_per_1000_people"] + (21 - x["nurses_and_midwives_per_1000_people"]) * (indicators["nm"] / 100), #* (1 + indicators["nm"] / 100),
        physicians_per_1000_people = lambda x: x["physicians_per_1000_people"] + (9 - x["physicians_per_1000_people"]) * (indicators["phys"] / 100),      
        prevalence_of_undernourishment = lambda x: x["prevalence_of_undernourishment"] - (x["prevalence_of_undernourishment"] - 0) * (indicators["undernourishment"] / 100), 
        share_of_population_urban = lambda x: x["share_of_population_urban"] + (100 - x["share_of_population_urban"]) * (indicators["urban"] / 100),
        share_without_improved_water = lambda x: x["share_without_improved_water"] - (x["share_without_improved_water"] - 0) * (indicators["water"] / 100),      
        vaccination_coverage_who_unicef = lambda x: x["vaccination_coverage_who_unicef"] + (100 - x["vaccination_coverage_who_unicef"]) * (indicators["vaccination"] / 100),    
        years_of_schooling = lambda x: x["years_of_schooling"] + (14 - x["years_of_schooling"]) * (indicators["school"] / 100)
    )
    #NEW_TEST
    modified_global_df = orig_global_df.assign(
        annual_healthcare_expenditure_per_capita = lambda x: x["annual_healthcare_expenditure_per_capita"] * (1 + indicators["ahec"] / 100),
        gdp_per_capita_worldbank = lambda x: x["gdp_per_capita_worldbank"] * (1 + indicators["gdp"] / 100),
        
        nurses_and_midwives_per_1000_people = lambda x: x["nurses_and_midwives_per_1000_people"] + (21 - x["nurses_and_midwives_per_1000_people"]) * (indicators["nm"] / 100), #* (1 + indicators["nm"] / 100),
        physicians_per_1000_people = lambda x: x["physicians_per_1000_people"] + (9 - x["physicians_per_1000_people"]) * (indicators["phys"] / 100),      
        prevalence_of_undernourishment = lambda x: x["prevalence_of_undernourishment"] - (x["prevalence_of_undernourishment"] - 0) * (indicators["undernourishment"] / 100), 
        share_of_population_urban = lambda x: x["share_of_population_urban"] + (100 - x["share_of_population_urban"]) * (indicators["urban"] / 100),
        share_without_improved_water = lambda x: x["share_without_improved_water"] - (x["share_without_improved_water"] - 0) * (indicators["water"] / 100),      
        vaccination_coverage_who_unicef = lambda x: x["vaccination_coverage_who_unicef"] + (100 - x["vaccination_coverage_who_unicef"]) * (indicators["vaccination"] / 100),    
        years_of_schooling = lambda x: x["years_of_schooling"] + (14 - x["years_of_schooling"]) * (indicators["school"] / 100)
    )
    #modified_global_df
    #NEW_TEST
    
    # ----------------------------------
    # Q-MODELS PREDICTIONS WITH ORIGINAL DF
    #----------------------------------- 
    X_original = years_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme", "pred_low", "pred_med", "pred_high", "q05_pos", "q075_pos", "bandwidth", "bandwidth_pos"])
    predicts_original = years_df.assign(
        pred_low  = qr_models["low"].predict(X_original) + SHIFT["q25"],
        pred_med  = qr_models["med"].predict(X_original) + SHIFT["q50"],
        pred_high = qr_models["high"].predict(X_original) + SHIFT["q75"]
    )
    
    # ----------------------------------
    # Q-MODELS PREDICTIONS WITH NEW DF
    #----------------------------------- 
    X_new = modified_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme", "pred_low", "pred_med", "pred_high", "q05_pos", "q075_pos", "bandwidth", "bandwidth_pos"])
    predicts_new = modified_df.assign(
        pred_low  = qr_models["low"].predict(X_new) + SHIFT["q25"],
        pred_med  = qr_models["med"].predict(X_new) + SHIFT["q50"],
        pred_high = qr_models["high"].predict(X_new) + SHIFT["q75"]
    )
    
    # ----------------------------------
    # Q-MODELS PREDICTIONS FOR OTHERS
    #----------------------------------- 
    X_orig_others = orig_global_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme", "pred_low", "pred_med", "pred_high", "q05_pos", "q075_pos", "bandwidth", "bandwidth_pos"])
    predicts_orig_global = orig_global_df.assign(
        pred_low  = qr_models["low"].predict(X_orig_others) + SHIFT["q25"],
        pred_med  = qr_models["med"].predict(X_orig_others) + SHIFT["q50"],
        pred_high = qr_models["high"].predict(X_orig_others) + SHIFT["q75"]
    )  
    X_new_others = modified_global_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme", "pred_low", "pred_med", "pred_high", "q05_pos", "q075_pos", "bandwidth", "bandwidth_pos"])
    predicts_new_global = modified_global_df.assign(
        pred_low  = qr_models["low"].predict(X_new_others) + SHIFT["q25"],
        pred_med  = qr_models["med"].predict(X_new_others) + SHIFT["q50"],
        pred_high = qr_models["high"].predict(X_new_others) + SHIFT["q75"]
    )  
    st.divider()
    # ----------------------------------
    # ___________PART 1___________
    # SHOW REFERENCE Q-MODELS PREDICTIONS 
    # WITHOUT ANY SIMULATION
    # PLUS QUANTILE FOCUS
    #----------------------------------- 

    q05_pos = predicts_original["q05_pos"].min()
    bw_med = predicts_original["bandwidth"].min()
    bw_med_pos = predicts_original["bandwidth_pos"].min()
    
    st.markdown(f"##### :orange[{base_df['Entity'].iloc[0]}'s] reference predicted child mortality rate for {' | '.join(map(str, sorted(years_select)))}")
    st.write(f"Based on a global reference, the corresponding quantile prediction specific to {base_df['Entity'].iloc[0]} in comparison to the remaining data is highlighted.")
    st.write("For convenience, each quantile reference prediction shows the smallest U5MR value if multiple years are chosen. The charts below show the predictions of the chosen years.")
    st.markdown(f"###### Without any indicator adjustments the U5MR prediction for {base_df['Entity'].iloc[0]} is:")
    
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
    #prevent quantile crossing
    q25_pred = np.minimum(predicts_original['pred_low'], predicts_original['pred_med'] - (1e-6))
    q50_pred = np.maximum(predicts_original['pred_med'], predicts_original['pred_low'] + (1e-6))
    q75_pred = np.maximum(predicts_original['pred_high'], predicts_original['pred_med'] + (1e-6))
    with quant_base1:
        st.metric(
            label="*Q 0.25 prediction*", 
            value=f"{q25_pred.min():.2f} per 1000",
            chart_data=q25_pred.round(2),
            chart_type="line", 
        )
        st.write(f"In 75% of cases with comparable feature combinations, the true value is above {predicts_original['pred_low'].min():.2f} per 1000")
        if focus_quant_025:
            with st.container():
                st.info("Focus Quantile")
    with quant_base2:
        st.metric(
            label="*Q 0.5 (median) prediction*", 
            value=f"{q50_pred.min():.2f} per 1000", 
            chart_data=q50_pred.round(2),
            chart_type="line", 
        )
        st.write(f"In 50% of cases with comparable feature combinations, the true value is between {predicts_original['pred_low'].min():.2f} and {predicts_original['pred_high'].min():.2f} per 1000")   
        if focus_quant_05:
            with st.container():
                st.warning("Focus Quantile") 
    with quant_base3:
        st.metric(
            label="*Q 0.75 prediction*", 
            value=f"{q75_pred.min():.2f} per 1000", 
            chart_data=q75_pred.round(2),
            chart_type="line", 
        )
        st.write(f"In 75% of cases with comparable feature combinations, the true value is below {predicts_original['pred_high'].min():.2f} per 1000")
        if focus_quant_075:
            with st.container():
                st.error("Focus Quantile")
    # ----------------------------------
    # FOCUS QUANTILES EXPLAINED
    #-----------------------------------            
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
    

    # ----------------------------------
    # TABS: GLOBAL & LOCAL SHAP PLOT
    #----------------------------------- 
    st.space()
    tab_global, tab_local = st.tabs(["Context View", "Local View"])
    global_beeswarm = f":orange[How the features affect the predictions for {base_df['world_income_group'].iloc[-1]} similar to **{base_df['Entity'].iloc[0]}**]"
    local_waterfall = f":orange[How the features affect the prediction for one year data sample of **{base_df['Entity'].iloc[0]}**]"
    shap_by_region = df_ref.copy()
    shap_by_income = df_ref.loc[df_ref["world_income_group"] == X_original["world_income_group"].iloc[-1]]
     
    choice_years = years_df["Year"].tolist()
    choice_year_label = "Choose year to view the specific local features' influence on the prediction"
    
    if focus_quant_025:
        with tab_global:
            st.write(global_beeswarm)
            shap_plot(qr_models, shap_by_income, "low", "Focus Quantile 0.25")
            #shap_bar_plot(qr_models, shap_by_income, "low", "High Income (Focus Quantile 0.25)")
        with tab_local:
            st.write(local_waterfall)     
            year_choice_orig1 = st.selectbox(label=choice_year_label, options=choice_years, key="year_choice_orig1")
            shap_decision_plot(qr_models, X_original, "low", f"{base_df['Entity'].iloc[0]} (Focus Quantile 0.25)", predicts_original["pred_low"], choice_years.index(year_choice_orig1))
            #force_plot(qr_models, X_original, "low", f"{base_df['Entity'].iloc[0]} (Focus Quantile 0.25)")  
        
    elif focus_quant_05:
        with tab_global:
            st.write(global_beeswarm)
            shap_plot(qr_models, shap_by_income, "med", "Focus Quantile 0.5")
        with tab_local:
            st.write(local_waterfall)
            year_choice_orig2 = st.selectbox(label=choice_year_label, options=choice_years, key="year_choice_orig2")
            shap_decision_plot(qr_models, X_original, "med", f"{base_df['Entity'].iloc[0]} (Focus Quantile 0.5)", predicts_original["pred_med"], choice_years.index(year_choice_orig2))

    else:
        with tab_global:
            st.write(global_beeswarm)
            shap_plot(qr_models, shap_by_income, "high", "Focus Quantile 0.75")
        with tab_local:
            st.write(local_waterfall)
            year_choice_orig3 = st.selectbox(label=choice_year_label, options=choice_years, key="year_choice_orig3")
            shap_decision_plot(qr_models, X_original, "high", f"{base_df['Entity'].iloc[0]} (Focus Quantile 0.75)", predicts_original["pred_high"], choice_years.index(year_choice_orig3))   
    st.divider()       
    st.divider()
    
if st.session_state.simulate_btn and (years_select and country_select is not None):
    # ----------------------------------
    # ___________PART 2___________
    # SHOW Q-MODELS PREDICTIONS 
    # AFTER SIMULATION
    #----------------------------------- 
    st.markdown(f"##### After indicator adjustments, :orange[{base_df['Entity'].iloc[0]}'s] simulated prediction of U5MR for {' | '.join(map(str, sorted(years_select)))} might have been")
    
    #prevent quantile crossing
    q25_new = np.minimum(predicts_new['pred_low'], predicts_new['pred_med'] - (1e-6))
    q50_new = np.maximum(predicts_new['pred_med'], predicts_new['pred_low'] + (1e-6))
    q75_new = np.maximum(predicts_new['pred_high'], predicts_new['pred_med'] + (1e-6))
    
    quant_col1, quant_col2, quant_col3 = st.columns(3, border=True)  
    with quant_col1:
        st.metric(
            label="*Q 0.25 prediction* (Best Case)", 
            delta_color="inverse",
            value=f"{q25_new.min():.2f} per 1000", 
            delta=f"""{q25_new.min() - q25_pred.min():.2f} per 1000
                        ({ ((q25_new.min() - q25_pred.min()) / q25_pred.min() * 100):.2f} %)""",
            chart_data=q25_new.round(2),
            chart_type="line",
        )
        if focus_quant_025:
            with st.container():
                st.success("Focus Quantile")
    with quant_col2:
        st.metric(
            label="*Q 0.5 (median) prediction*", 
            delta_color="inverse",
            value=f"{q50_new.min():.2f} per 1000", 
            delta=f"""{q50_new.min() - q50_pred.min():.2f} per 1000
                    ({ ((q50_new.min() - q50_pred.min()) / q50_pred.min() * 100):.2f} %)""",
            chart_data=q50_new.round(2),
            chart_type="line",
        )
        if focus_quant_05:
            with st.container():
                st.success("Focus Quantile")  
    with quant_col3:
        st.metric(
            label="*Q 0.75 prediction* (Worst Case)", 
            delta_color="inverse",
            value=f"{q75_new.min():.2f} per 1000", 
            delta=f"""{q75_new.min() - q75_pred.min():.2f} per 1000
                    ({ ((q75_new.min() - q75_pred.min()) / q75_pred.min() * 100):.2f} %)""",
            chart_data=q75_new.round(2),
            chart_type="line",
        )
        if focus_quant_075:
            with st.container():
                st.success("Focus Quantile")
                
    st.space()
    tab_local_new = st.tabs(["Local View"])[0] 
    
    # ----------------------------------
    # SHOW SHAP PLOT  
    # AFTER SIMULATION
    #----------------------------------- 
    chart_by_income = df_ref.loc[df_ref["world_income_group"] == X_original["world_income_group"].iloc[-1]]
    
    with tab_local_new:
        st.markdown(f"###### :orange[How the adjusted features affect the new prediction]") 
        if focus_quant_025:
            year_choice_new1 = st.selectbox(label=choice_year_label, options=choice_years, key="year_choice_new1")
            shap_decision_plot(qr_models, X_new, "low", f"{base_df['Entity'].iloc[0]} (Focus Quantile 0.25)", predicts_new["pred_low"], choice_years.index(year_choice_new1))
        elif focus_quant_05:
            year_choice_new2 = st.selectbox(label=choice_year_label, options=choice_years, key="year_choice_new2")
            shap_decision_plot(qr_models, X_new, "med", f"{base_df['Entity'].iloc[0]} (Focus Quantile 0.5)", predicts_new["pred_med"], choice_years.index(year_choice_new2))
        else:
            year_choice_new3 = st.selectbox(label=choice_year_label, options=choice_years, key="year_choice_new3")
            shap_decision_plot(qr_models, X_new, "high", f"{base_df['Entity'].iloc[0]} (Focus Quantile 0.75)", predicts_new["pred_high"], choice_years.index(year_choice_new3))
            #st.line_chart(chart_by_income, x="", y="")
            #chart_by_income
     
    df_preds = pd.DataFrame({
    "base_q025": q25_pred,
    "whatif_q025": q25_new,
    "base_q05": q50_pred,
    "whatif_q05": q50_new,
    "base_q075": q75_pred,
    "whatif_q075": q75_new,
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


    # ----------------------------------
    # ___________PART 3___________
    # SHOW SENSITIVITY BY 1 FEATURE
    # BY INCOME GROUPS   
    # AFTER SIMULATION
    #-----------------------------------
    if "changed_sliders" in st.session_state and st.session_state.changed_sliders:
        st.divider()
        st.divider()
        st.markdown(f"##### Indicator Sensitivity on the simulated U5MR prediction for :orange[{base_df['Entity'].iloc[0]}] & *by Income Groups* | *by World Regions*")
        focusQ = ""
        
        if len(st.session_state.changed_sliders) > 1:
            st.info(f"Since more than one factor were changed [{', '.join(st.session_state.changed_sliders)}], the plots show the sensitivity taking into account all active sliders.")
    
        #Choose factor, group, effect type
        list_features = st.session_state.changed_sliders
        choose_factor = st.selectbox(label="Sensitivity of factor:", options=list_features, key="factor")
        abs_eff_col, rel_eff_col = st.columns(2)
        with abs_eff_col:
            choose_group = st.radio("Impact segmented by:", ["world_income_group", "world_regions_wb"], horizontal=True, key="group")
        with rel_eff_col:
            choose_effect = st.radio("Absolute vs. Relative sensitivity:", ["absolute impact (per 1000)", "relative impact (%)"], horizontal=True, key="effect")
                
        #Show Focus Quantile Scatterplot
        if focus_quant_025:
            focusQ = "pred_low"
            st.markdown("<h4 style='text-align: center;'>Focus Quantile: Q0.25</h4>", unsafe_allow_html=True)
        elif focus_quant_05:
            focusQ = "pred_med"
            st.markdown("<h4 style='text-align: center;'>Focus Quantile: Q0.5</h4>", unsafe_allow_html=True)
        else:
            focusQ = "pred_high"
            st.markdown("<h4 style='text-align: center;'>Focus Quantile: Q0.75</h4>", unsafe_allow_html=True)
        
        y_pred_orig_global = predicts_orig_global[focusQ]        
        y_pred_new_global =  predicts_new_global[focusQ]
        if choose_effect == "absolute impact (per 1000)":
            predicts_new_global[choose_effect] = y_pred_new_global - y_pred_orig_global   #absolut
        else:
            predicts_new_global[choose_effect] = ((y_pred_new_global - y_pred_orig_global) / y_pred_orig_global) * 100 #relativ
        country_med = predicts_new_global[predicts_new_global["Year"].isin(years_select)].groupby([choose_group, choose_factor, "Entity", focusQ])[choose_effect].mean().reset_index()        
        
        #Sensitivity Plot (Scatterplot)
        col_sens1, col_sens2, col_sens3 = st.columns([1, 10, 1])
        with col_sens2:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.scatterplot(
                x=country_med[choose_factor].round(1),
                y=country_med[choose_effect].round(4),
                hue=choose_group,
                s=100,
                data=country_med,
                ax=ax
            )
            ax.set_ylabel(f"{choose_effect} on U5MR")
            ax.set_title(f"Sensitvity of U5MR by {choose_factor}")
            for i in range(country_med.shape[0]):
                if country_med.Entity[i] == years_df["Entity"].iloc[0]:
                    plt.text(x=country_med[choose_factor][i]+0.1, 
                        y=country_med[choose_effect][i], 
                        s=country_med.Entity.iloc[i],   
                        fontdict=dict(color='black', size=10),
                        bbox=dict(facecolor='white', alpha=0.8))
                    break
            st.pyplot(fig)

            if choose_group == "world_income_group":
                fig_marg_eff = sns.lmplot(data=country_med, x=choose_factor, y=focusQ, 
                                col="world_income_group", hue="world_income_group", col_wrap=2, height=4, aspect=1.78)
                fig_marg_eff.set_titles(template="{col_name}")
                fig_marg_eff.fig.suptitle(f"Conditional Effects of {choose_factor} on U5MR by {choose_group}", fontsize=18, y=1.05)
                fig_marg_eff.set_axis_labels(choose_factor, "U5MR per 1000")
                st.pyplot(plt.gcf())  
                
                #Compare with other countries
                st.multiselect(label=f"Compare {years_df['Entity'].iloc[0]} with other countries",
                                options=df_ref["Entity"].unique().tolist(),
                                max_selections=2,
                                placeholder="Countries")  
    #-------------- 
    #--------------   
    #--------------   
    #--------------   
    #--------------   
    #--------------   
    #--------------     
    #st.info(f"ORIGINAL PRED (25%, 50%, 75%): "
    #    f"{predicts_original['pred_low'].median():.2f}, {predicts_original['pred_med'].median():.2f}, {predicts_original['pred_high'].median():.2f}")
    #st.info(f"NEW PRED (25%, 50%, 75%): "
    #    f"{predicts_new['pred_low'].tolist()}, {predicts_new['pred_med'].median():.2f}, {predicts_new['pred_high'].median():.2f}")
    st.divider()
    expander = st.expander(f"For reference, view actual historical child mortality rates for {base_df['Entity'].iloc[0]} (ground truth)")
    for i, (idx, row) in enumerate(years_df.iterrows()):        
        expander.write(f'{row["Year"]}: {row["child_mortality_igme"]:.2f} per 1000')
    