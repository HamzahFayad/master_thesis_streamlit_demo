import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_models():
    return {
        "low": joblib.load("model/quantile_0.25.pkl"),    
        "med": joblib.load("model/quantile_0.5.pkl"),   
        "high": joblib.load("model/quantile_0.75.pkl")
    } 
      
@st.cache_data
def load_df():
    df = pd.read_csv("reference_data/base_df.csv")
    return df
 

def build_sidebar(years_df, years_select, country_select):
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
            min_value=-float(years_df['share_without_improved_water'].min()) if not years_df['share_without_improved_water'].median() <= 0.1 else -1.0,
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