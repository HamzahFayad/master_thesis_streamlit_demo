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
        "low": joblib.load("model/quantile_025.pkl"),    
        "med": joblib.load("model/quantile_05.pkl"),   
        "high": joblib.load("model/quantile_075.pkl")
    } 
# ----------------------------------
# REFERENCE DATASET 
#-----------------------------------     
@st.cache_data
def load_df():
    df = pd.read_csv("reference_data/base_df.csv")
    return df
 
 

# ----------------------------------
# SIDEBAR WITH FEATURE ADJUSTMENTS 
#----------------------------------- 
slider_vars = {
    "ahec": 0,
    "gdp": 0, 
    "nm": 0, 
    "phys": 0, 
    "vaccination": 0, 
    "urban": 0, 
    "undernourishment": 0, 
    "water": 0,
    "school": 0    
}
def build_sidebar(years_df, years_select, country_select):
    with st.sidebar:
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
            "Increase :orange[*nurses/midwives per 1000 people*]",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            format="%.1f%%",
            help="Nurses and midwives include professional nurses, professional midwives, auxiliary nurses & midwives, enrolled nurses & midwives and other associated personnel."
            )
            st.caption(
            f"~ {(years_df['nurses_and_midwives_per_1000_people'].median() * (1 + slider_vars['nm'] / 100)):.2f} per 1000"
            )
            st.space()
            
            #Physicians per 1000
            slider_vars["phys"] = st.slider(
            "Increase :orange[*physicians per 1000 people*]",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            format="%.1f%%",
            help="Physicians include generalist and specialist medical practitioners."
            )
            st.caption(
            f"~ {(years_df['physicians_per_1000_people'].median() * (1 + slider_vars['phys'] / 100)):.2f} per 1000"
            )
            st.space()
            
            #vaccination coverage 
            slider_vars["vaccination"] = st.slider(
            "Increase :orange[*vaccination coverage*]",
            min_value=0.0,
            max_value=100 - float(years_df["vaccination_coverage_who_unicef"].max()),
            value=0.0,
            step=0.5,
            format="%.1f%%",
            help="Share of one-year-olds who have had three doses of the combined diphtheria, tetanus and pertussis vaccine in a given year."
            )
            st.caption(
            f"~ {( min(100.0, years_df['vaccination_coverage_who_unicef'].median() + slider_vars['vaccination']) ):.2f} %"
            )
            st.space()
            
            #-------------------------------------------------
            st.markdown("#### Living Standards:")
            
            #Share of population urban 
            slider_vars["urban"] = st.slider(
            "Increase :orange[*share of population urban*]",
            min_value=0.0,
            max_value=100 - float(years_df["share_of_population_urban"].max()) if not years_df['share_of_population_urban'].median() >= 100.0 else 0.5,
            #max_value=float(100.0 / years_df["share_of_population_urban"].median() -1) * 100, #50.0,
            value=0.0,
            step=0.5,
            format="%.1f%%",
            disabled=years_df['share_of_population_urban'].median() >= 100.0,
            help="Share of the population living in urban areas."
            )
            st.caption(
            f"~ {( min(100.0, years_df['share_of_population_urban'].median() + slider_vars['urban']) ):.2f} %"
            )
            st.space()
            
            #Prevalence of undernourishment
            slider_vars["undernourishment"] = st.slider(
            "Decrease :orange[*prevalence of undernourishment*]",
            #min_value=-100.0,
            min_value=-float(years_df['prevalence_of_undernourishment'].min()),
            max_value=0.0,
            value=0.0,
            step=0.5,
            disabled=years_df['prevalence_of_undernourishment'].median() <= 0.1,
            format="%.1f%%",
            help="Share of the population whose daily food intake does not provide enough energy to maintain a normal, active, and healthy life."
            )
            st.caption(
            f"~ {(years_df['prevalence_of_undernourishment'].median() + slider_vars['undernourishment']):.2f} %"
            )
            st.space()

            #Share without improved water 
            slider_vars["water"] = st.slider(
            "Decrease :orange[*share of population without improved water*]",
            #min_value=-float(years_df['share_without_improved_water'].min()) if not years_df['share_without_improved_water'].median() <= 0.1 else -1.0,
            min_value=-float(years_df['share_without_improved_water'].median()) if not years_df['share_without_improved_water'].median() <= 0.1 else -1.0,
            max_value=0.0,
            value=0.0,
            step=0.5,
            disabled=years_df['share_without_improved_water'].median() <= 0.1,
            format="%.1f%%",
            help="Improved drinking water sources are those that have the potential to deliver safe water by nature of their design and construction, and include: piped water, boreholes or tubewells, protected dug wells, protected springs, rainwater, and packaged or delivered water."
            )
            st.caption(
            f"~ {( years_df['share_without_improved_water'].median() + slider_vars['water'] ):.2f} %"
            )
            st.space()
            
            #-------------------------------------------------
            st.markdown("#### Education:")
            
            #years of schooling 
            slider_vars["school"] = st.slider(
            "Increase :orange[*years of schooling*]",
            min_value=0.0,
            max_value=float(max(0.0, 14.0 - years_df['years_of_schooling'].median())) if not years_df['years_of_schooling'].median() > 14.0 else 0.5,
            value=0.0,
            step=0.5,
            disabled=years_df['years_of_schooling'].median() > 14.0,
            format="%.1f years",
            help="Average number of years women aged 25 and older have spent in formal education."
            )
            st.caption(
            f"~ {( years_df['years_of_schooling'].median() + slider_vars['school']):.1f} school years"
            )
            st.space()
            
            st.divider()
            if st.button("Simulate", type="primary"):
                st.session_state.simulate_btn = True
        
    return slider_vars
 

# ----------------------------------
# SHAP SUMMARY PLOT 
#----------------------------------- 
def rename_features(feature_names):
    rename_features = {
    "nurses_and_midwives_per_1000_people": "nurses/midwives per 1000",
    "physicians_per_1000_people": "physicians per 1000",
    "prevalence_of_undernourishment": "prevalence of undernourishment",
    "share_of_population_urban": "share of population urban",
    "share_without_improved_water": "share without improved water", 
    "vaccination_coverage_who_unicef": "vaccination coverage",
    "years_of_schooling": "years of schooling"       
    }
    prefix = ["world_regions_wb_", "world_income_group_"]
    
    new_feture_names = []
    for n in feature_names:
        for p in prefix:
            n = n.removeprefix(p)
        final_name = rename_features.get(n, n)
        new_feture_names.append(final_name)
        
    return new_feture_names

@st.cache_data
def setup_shap(_model, _data):
    return shap.LinearExplainer(_model, _data)

def shap_plot(qr_models, X, quant, title):
    inner_pipeline = qr_models[quant].regressor_
    preprocessor = inner_pipeline.named_steps["preprocess"]
    model = inner_pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X) 
    #X_transformed = np.exp(X_transformed)
    feature_names = preprocessor.get_feature_names_out()
    
    new_feature_names = rename_features(feature_names)

    expl = setup_shap(model, X_transformed)
    shapvals = expl(X_transformed)
    shapvals.feature_names = list(new_feature_names)
    
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        #shap.plots.bar(shapvals.abs.sum(0))
        shap.summary_plot(shapvals, X_transformed, feature_names=new_feature_names, plot_size=[12,6], show=False)
        #shap.plots.waterfall(shapvals[4])
        plt.title(f"Feature Impact on the Prediction: {title}")
        st.pyplot(fig)
        plt.clf()
        plt.close()
      

def shap_decision_plot(qr_models, X, quant, title):
    inner_pipeline = qr_models[quant].regressor_
    preprocessor = inner_pipeline.named_steps["preprocess"]
    model = inner_pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X) 
    feature_names = preprocessor.get_feature_names_out()
    
    new_feature_names = rename_features(feature_names)

    expl = setup_shap(model, X_transformed)
    shapvals = expl(X_transformed)    
    shapvals.feature_names = list(new_feature_names)
    
    """exp = shap.Explanation(
        values=shapvals.values[0].reshape(1, -1),
        base_values=shapvals.base_values[0],
        data=X_transformed[0].reshape(1, -1), 
        feature_names=new_feature_names
    )"""

    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        #shap.initjs()
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.plots.waterfall(shapvals[0], max_display=10, show=False)
        
        for ax in fig.axes:
            ax.set_xlabel("") 
            ax.set_xticks([])
            for text in ax.texts:
                t = text.get_text()
                if "=" in t:
                    new_text = t.split("=")[-1].strip()
                    text.set_text(new_text)
                if "f(x)" in t or "E[f(x)]" in t:
                    text.set_text("")

        ax.set_yticklabels([label.get_text().split('=')[-1].strip() if '=' in label.get_text() else label.get_text() for label in ax.get_yticklabels()])
        #shap.plots.scatter(shapvals[:, "share without improved water"], color=shapvals[:, 0], ax=ax, show=False)
        #st.pyplot(fig)
        #plt.title(f"Feature Impact on the Prediction: {title}")
        st.pyplot(fig)
        plt.clf()
        plt.close()
