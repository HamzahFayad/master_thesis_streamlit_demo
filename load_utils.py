import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import shap

# ----------------------------------
# MODELS 
#----------------------------------- 
@st.cache_resource 
def load_models():
    return {
        "low": joblib.load("model_final/int_quantile_025.pkl"),    
        "med": joblib.load("model_final/int_quantile_05.pkl"),   
        "high": joblib.load("model_final/int_quantile_075.pkl")
    } 
"""    
def load_models(income):
    print(income)
    return {
        "low": joblib.load(f"model/{income}_quantiles_new_0.25.pkl"),    
        "med": joblib.load(f"model/{income}_quantiles_new_0.5.pkl"),   
        "high": joblib.load(f"model/{income}_quantiles_new_0.75.pkl")
    } 
"""
# ----------------------------------
# REFERENCE DATASET 
#-----------------------------------     
@st.cache_data
def load_df():
    #df = pd.read_csv("reference_data/base_df.csv")
    df = pd.read_csv("reference_data/reference_df.csv")
    return df
 
 
#Features
DEFAULTS = {
    "nurses_and_midwives_per_1000_people": 0.0, 
    "physicians_per_1000_people": 0.0, 
    "prevalence_of_undernourishment": 0.0, 
    "share_of_population_urban": 0.0, 
    "share_without_improved_water": 0.0, 
    "vaccination_coverage_who_unicef": 0.0,
    "years_of_schooling": 0.0    
}

# ----------------------------------
# SIDEBAR WITH FEATURE ADJUSTMENTS 
#----------------------------------- 
slider_vars = {
    "ahec": 0.0,
    "gdp": 0.0, 
    "nm": 0.0, 
    "phys": 0.0, 
    "vaccination": 0.0, 
    "urban": 0.0, 
    "undernourishment": 0.0, 
    "water": 0.0,
    "school": 0.0    
}

@st.fragment
def build_sidebar(years_df, years_select, country_select):
    if years_select and country_select is not None:
        st.divider()
        st.subheader(f"Adjust indicators to simulate hypothetical scenarios of child mortality rate in: :orange[*{country_select}*] \u2193")
        st.markdown('<h5 style="font-weight: 400;"><em>(changes affect all selected years)</em></h5>', unsafe_allow_html=True)
        st.space()
            
        #-------------------------------------------------
        st.markdown("#### Economic Performance & Welfare:")

        #Annual Healthcare Exp. per capita
        slider_vars["ahec"] = st.slider(
        "Increase :orange[*annual health spending per capita*]",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        format="%.1f%%",
        help="The sum of public and private annual health expenditure per person. This data is adjusted for differences in living costs between countries, but it is not adjusted for inflation."
        )
        st.caption(
        f"~ {(years_df['annual_healthcare_expenditure_per_capita'].median() * (1 + slider_vars['ahec'] / 100)):.2f} int. $"
        )
        st.space()
            
        #GDP per capita
        slider_vars["gdp"] = st.slider(
        "Increase :orange[*gross domestic product (GDP) per capita*]",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        format="%.1f%%",
        help="Average economic output per person in a country or region per year. This data is adjusted for inflation and differences in living costs between countries."
        )
        st.caption(
        f"~ {(years_df['gdp_per_capita_worldbank'].median() * (1 + slider_vars['gdp'] / 100)):.2f} int. $, PPP"
        )  
        st.space()
            
        #-------------------------------------------------
        st.markdown("#### Medical Health:")
        
        #Nurses & midwives per 1000
        slider_vars["nm"] = st.slider(
        "Increase :orange[*nurses/midwives per 1000 people*] (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        format="%.1f%%",
        key="nurses_and_midwives_per_1000_people",
        help="Nurses and midwives include professional nurses, professional midwives, auxiliary nurses & midwives, enrolled nurses & midwives and other associated personnel."
        )
        current_val_nm = years_df['nurses_and_midwives_per_1000_people'].median()
        new_val_nm = current_val_nm + (21 - current_val_nm) * (slider_vars["nm"] / 100)
        st.caption(
        f"**current**: {(current_val_nm):.2f} per 1000 | **new**: {(new_val_nm):.2f} per 1000"
        )
        st.space()
            
        #Physicians per 1000
        slider_vars["phys"] = st.slider(
        "Increase :orange[*physicians per 1000 people*] (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        format="%.1f%%",
        key="physicians_per_1000_people",
        help="Physicians include generalist and specialist medical practitioners."
        )
        current_val_ph = years_df['physicians_per_1000_people'].median()
        new_val_phys = current_val_ph + (9 - current_val_ph) * (slider_vars["phys"] / 100)
        st.caption(
        f"**current**: {(current_val_ph):.2f} per 1000 | **new**: {(new_val_phys):.2f} per 1000"
        )
        st.space()
            
        #vaccination coverage 
        slider_vars["vaccination"] = st.slider(
        "Increase :orange[*vaccination coverage*] (%)",
        min_value=0.0,
        max_value=100.0, #- float(years_df["vaccination_coverage_who_unicef"].max()),
        value=0.0,
        step=1.0,
        disabled=years_df['vaccination_coverage_who_unicef'].median() >= 99.9,
        format="%.1f%%",
        key="vaccination_coverage_who_unicef",
        help="Share of one-year-olds who have had three doses of the combined diphtheria, tetanus and pertussis vaccine in a given year."
        )
        current_val_vacc = years_df['vaccination_coverage_who_unicef'].median()
        new_val_vacc = current_val_vacc + (100 - current_val_vacc) * (slider_vars["vaccination"] / 100)
        st.caption(
        f"**current**: {(current_val_vacc):.2f} % | **new**: {(new_val_vacc):.2f} %"
        )
        st.space()
            
        #-------------------------------------------------
        st.markdown("#### Living Standards:")
            
        #Share of population urban 
        slider_vars["urban"] = st.slider(
        "Increase :orange[*share of population urban*] (%)",
        min_value=0.0,
        max_value=100.0, #- float(years_df["share_of_population_urban"].max()) if not years_df['share_of_population_urban'].median() >= 100.0 else 0.5,
        #max_value=float(100.0 / years_df["share_of_population_urban"].median() -1) * 100, #50.0,
        value=0.0,
        step=1.0,
        format="%.1f%%",
        key="share_of_population_urban",
        disabled=years_df['share_of_population_urban'].median() >= 99,
        help="Share of the population living in urban areas."
        )
        current_val_urban = years_df['share_of_population_urban'].median()
        new_val_urban = current_val_urban + (100 - current_val_urban) * (slider_vars["urban"] / 100)
        st.caption(
        f"**current**: {(current_val_urban):.2f} % | **new**: {(new_val_urban):.2f} %"
        )
        st.space()
            
        #Prevalence of undernourishment
        slider_vars["undernourishment"] = st.slider(
        "Decrease :orange[*prevalence of undernourishment*] (%)",
        #min_value=-100.0,
        min_value=0.0, #-float(years_df['prevalence_of_undernourishment'].min()),
        max_value=100.0,
        value=0.0,
        step=1.0,
        disabled=years_df['prevalence_of_undernourishment'].median() <= 0.1,
        format="%.1f%%",
        key="prevalence_of_undernourishment",
        help="Share of the population whose daily food intake does not provide enough energy to maintain a normal, active, and healthy life."
        )
        current_val_und = years_df['prevalence_of_undernourishment'].median()
        new_val_und = current_val_und - (current_val_und - 0) * (slider_vars["undernourishment"] / 100)
        st.caption(
        f"**current**: {(current_val_und):.2f} % | **new**: {(new_val_und):.2f} %"
        )
        st.space()

        #Share without improved water 
        slider_vars["water"] = st.slider(
        "Decrease :orange[*share of population without improved water*] (%)",
        #min_value=-float(years_df['share_without_improved_water'].min()) if not years_df['share_without_improved_water'].median() <= 0.1 else -1.0,
        min_value= 0.0, #-float(years_df['share_without_improved_water'].median()) if not years_df['share_without_improved_water'].median() <= 0.1 else -1.0,
        max_value=100.0,
        value=0.0,
        step=1.0,
        disabled=years_df['share_without_improved_water'].median() <= 0.1,
        format="%.1f%%",
        key="share_without_improved_water",
        help="Improved drinking water sources are those that have the potential to deliver safe water by nature of their design and construction, and include: piped water, boreholes or tubewells, protected dug wells, protected springs, rainwater, and packaged or delivered water."
        )
        current_val_water = years_df['share_without_improved_water'].median()
        new_val_water = current_val_water - (current_val_water - 0) * (slider_vars["water"] / 100)
        st.caption(
        f"**current**: {(current_val_water):.2f} % | **new**: {(new_val_water):.2f} %"
        )
        st.space()
            
        #-------------------------------------------------
        st.markdown("#### Education:")
            
        #years of schooling 
        slider_vars["school"] = st.slider(
        "Increase :orange[*years of schooling*] (%)",
        min_value=0.0,
        max_value=100.0, #float(max(0.0, 14.0 - years_df['years_of_schooling'].median())) if not years_df['years_of_schooling'].median() > 14.0 else 0.5,
        value=0.0,
        step=1.0,
        disabled=years_df['years_of_schooling'].median() >= 13.0,
        format="%.1f%%",
        key="years_of_schooling",
        help="Average number of years women aged 25 and older have spent in formal education."
        )
        current_val_school = years_df['years_of_schooling'].median()
        new_val_school = current_val_school + (13 - current_val_school) * (slider_vars["school"] / 100)
        st.caption(
        f"**current**: {(current_val_school):.1f} school years | **new**: {(new_val_school):.1f} school years"
        )
        st.space()
            
        st.divider()
        if st.button("Simulate", type="primary"):
            changed_sliders = [
                key for key, default in DEFAULTS.items()
                if st.session_state.get(key) != default
            ]
            st.session_state.changed_sliders = changed_sliders
            st.session_state.simulate_btn = True
            st.rerun()
        
    return slider_vars

# ----------------------------------
# FEATURES FOR ORIGINAL VALUES 
#----------------------------------- 
def rename_features(feature_names):
    rename_features = {
    #"annual_healthcare_expenditure_per_capita": "annual_healthcare_expenditure_per_capita",
    #"gdp_per_capita_worldbank": "gdp_per_capita_worldbank",
    "healthspending_gdp_ratio": "healthspending_gdp_ratio",
    "nurses_and_midwives_per_1000_people": "nurses_and_midwives_per_1000_people",
    "physicians_per_1000_people": "physicians_per_1000_people",
    "prevalence_of_undernourishment": "prevalence_of_undernourishment",
    "share_of_population_urban": "share_of_population_urban",
    "share_without_improved_water": "share_without_improved_water", 
    "vaccination_coverage_who_unicef": "vaccination_coverage_who_unicef",
    "years_of_schooling": "years_of_schooling",
    #"health_gdp_ratio_x_world_income_group_Upper-middle-income countries": "healthspending_gdp_ratio * upper-middle-income countries"    
    }
    prefix = ["world_regions_wb_", "world_income_group_"]
    
    new_feture_names = []
    for n in feature_names:
        for p in prefix:
            n = n.removeprefix(p)
        final_name = rename_features.get(n, n)
        new_feture_names.append(final_name)
        
    return new_feture_names

# ----------------------------------
# CREATE SHAP EXPLAINER
#----------------------------------- 
@st.cache_resource
def setup_shap(_model, _data):
    return shap.Explainer(_model, _data)

# ----------------------------------
# GENERATE SHAP VALUES BY FEATURES
# + FEATURE NAMES LIST
#----------------------------------- 
def create_shap_by_models(qr_models, X, quant):
    inner_pipeline = qr_models[quant].regressor_
    preprocessor = inner_pipeline.named_steps["preprocess"]
    model = inner_pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X) 
    feature_names = preprocessor.get_feature_names_out() 
    new_feature_names = rename_features(feature_names)
    
    expl = setup_shap(model, X_transformed)
    shapvals = expl(X_transformed)    
    shapvals.feature_names = list(new_feature_names)
    #print(new_feature_names)
    #bv_expanded = shapvals.base_values[:, None]
    #shapvals.values = np.expm1(shapvals.values + bv_expanded) - np.expm1(bv_expanded)
    #shapvals.base_values = np.expm1(shapvals.base_values)
    
    return X_transformed, new_feature_names, shapvals

# ----------------------------------
# GLOBAL BEESWARM SHAP PLOT
#----------------------------------- 
def shap_plot(qr_models, X, quant, title):
    
    X_transformed, new_feature_names, shapvals = create_shap_by_models(qr_models, X, quant)
 
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        #shap.plots.bar(shapvals.abs.sum(0))
        #shap.plots.waterfall(shapvals[4])
        shap.summary_plot(shapvals, X_transformed, feature_names=new_feature_names, 
                          plot_size=[12,6], max_display=15, show=False)
        plt.title(f"Features Impact on the Prediction: {title}")
        st.pyplot(fig)
        plt.clf()
        plt.close()

# ----------------------------------
# LOCAL WATERFALL SHAP PLOT
#-----------------------------------       
def shap_decision_plot(qr_models, X, quant, title, prediction, id):
    
    X_transformed, new_feature_names, shapvals = create_shap_by_models(qr_models, X, quant)

    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        #shap.initjs()
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.plots.waterfall(shapvals[id], max_display=15, show=False)
         
        for ax in fig.axes:
            ax.set_xlabel("")  
            ax.set_xticks([])
        ax.set_xlabel(f"f(x) = {prediction.iloc[id]:.2f}")
        
        row_idx = id
        orig_row = X.iloc[row_idx]
        new_labels = []
        current_yticklabels = ax.get_yticklabels()

        for label in current_yticklabels:
            text = label.get_text()  
            if '=' in text and "healthspending_gdp_ratio" in text:
                ratio_name = "healthspending_gdp_ratio"
                ratio = orig_row["annual_healthcare_expenditure_per_capita"] / orig_row["gdp_per_capita_worldbank"]
                formatted_ratio = f"{ratio:,.2f}"
                new_labels.append(f"{formatted_ratio} = {ratio_name}") 
            #elif '=' in text and "urban_x_medical_access" in text:
            #    um_name = "urban_x_medical_access"
            #    urban_medical = (orig_row["nurses_and_midwives_per_1000_people"] + orig_row["physicians_per_1000_people"]) * orig_row["share_of_population_urban"]
            #    formatted_um = f"{urban_medical:,.2f}"
            #   new_labels.append(f"{formatted_um} = {um_name}") 
                #new_labels.remove(text)
            elif '=' in text:
                txt_parts = text.split('=')
                name_part = txt_parts[-1].strip()         
                clean_name = name_part.split('__')[-1]
                
                if clean_name in orig_row:
                    real_value = orig_row[clean_name]
                    
                    if isinstance(real_value, (int, float)):
                        formatted_val = f"{real_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                        if formatted_val.endswith(",00"):
                            formatted_val = formatted_val[:-3]
                    else:
                        formatted_val = str(real_value)

                    new_labels.append(f"{formatted_val} = {clean_name}")
                else:
                    new_labels.append(text)
            else:
                new_labels.append(text)

        ax.set_yticklabels(new_labels)
        #ax.set_yticklabels([label.get_text().split('=')[-1].strip() if '=' in label.get_text() else label.get_text() for label in ax.get_yticklabels()])
        plt.title(f"Features Impact on one single prediction: {title}")
        st.pyplot(fig)
        plt.clf()
        plt.close()

# ----------------------------------
# BAR PLOT BY INCOME GROUP
#-----------------------------------  

def shap_bar_plot(qr_models, X, quant, title):
        
    X_transformed, new_feature_names, shapvals = create_shap_by_models(qr_models, X, quant)
    
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        #shap.summary_plot(shapvals, X_transformed, feature_names=new_feature_names, 
        #                  plot_size=[12,6], max_display=24, show=False)  
        shap.dependence_plot("years_of_schooling", shapvals.values, X_transformed)      
        plt.title(f"Features Impact on the Prediction: {title}")
        st.pyplot(fig) 
        plt.clf()
        plt.close()

# ----------------------------------
# SHAP FORCE PLOT
#-----------------------------------  
import streamlit.components.v1 as components
def st_shap(plot, height=None):
    shap_html = f"""
    <head>
        {shap.getjs()}
        <style>
            body {{
                background-color: white !format;
                color: black !important;
            }}
        </style>
    </head>
    <body>
        <div style="background-color: white; padding: 10px; border-radius: 5px;">
            {plot.html()}
        </div>
    </body>
    """
    components.html(shap_html, height=height)
    
def force_plot(qr_models, X, quant, title):
    X_transformed, new_feature_names, shapvals = create_shap_by_models(qr_models, X, quant)
    force_p = shap.plots.force(shapvals[0:150])
    st_shap(force_p, height=400)


# ----------------------------------
# SHAP DEPENDANCE PLOT
#-----------------------------------  
def dependance_plot(qr_models, X, quant, title):
    X_transformed, new_feature_names, shapvals = create_shap_by_models(qr_models, X, quant)
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        idx = list(X.columns).index("years_of_schooling")
        shap.plots.scatter(shapvals[:, idx])
        plt.title(f"Features Impact on the Prediction: {title}")
        st.pyplot(fig)
        plt.clf()
        plt.close()
