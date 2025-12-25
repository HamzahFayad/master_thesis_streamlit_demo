import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import joblib

from utils import ratio_health_gdp

# ----------------------------------
# LOAD MODELS
#-----------------------------------

@st.cache_resource
def load_models():
    return {
        "low": joblib.load("./model/quantile_0.25.pkl"),    
        "med": joblib.load("./model/quantile_0.5.pkl"),   
        "high": joblib.load("./model/quantile_0.75.pkl")
    }

qr_models = load_models()


# ----------------------------------
# LOAD REFERENCE DATAFRAME
#-----------------------------------
@st.cache_data
def load_df():
    df = pd.read_csv("./reference_data/base_dataset.csv")
    return df

df_ref = load_df()

# ----------------------------------
# CHOOSE COUNTRY & YEARS
#-----------------------------------
country_select = st.sidebar.selectbox(
    label="Select a country",
    options=df_ref["Entity"].unique().tolist(),
    index=None,
    placeholder="Country",
)
base_df = df_ref[df_ref["Entity"] == country_select].copy()

years_select = st.sidebar.multiselect(
    label="Select Year",
    options=base_df["Year"].tolist(),
    default=[],
    placeholder="Available Years",
)
years_df = base_df[base_df["Year"].isin(years_select)].copy()

if country_select is not None:
    st.title(base_df["Entity"].iloc[0])

 
# ----------------------------------
# NEW CHANGES DF TO PREDICT WITH
#-----------------------------------
modified_df = years_df.copy()

with st.sidebar:
    if years_select and country_select is not None:
        st.divider()
        st.subheader(f"Adjust indicators to simulate hypothetical scenarios of child mortality rate in: :orange[*{country_select}*]")
        st.space()
        
        #Annual Healthcare Exp. per capita
        ahec = st.slider(
        "Increase :orange[*annual health spending per capita*]",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=1.0,
        format="%.1f%%",
        help="The sum of public and private annual health expenditure per person. This data is adjusted for differences in living costs between countries, but it is not adjusted for inflation."
        )
        st.caption(
        f"~ {(years_df['annual_healthcare_expenditure_per_capita'].mean() * (1 + ahec / 100)):.2f} int. $"
        )
        st.space()
        
        #GDP per capita
        #gdp = st.slider(
        #"Increase :orange[*gross domestic product (GDP) per capita*]",
        #min_value=0.0,
        #max_value=50.0,
        #value=0.0,
        #step=1.0,
        #format="%.1f%%",
        #help="Average economic output per person in a country or region per year. This data is adjusted for inflation and differences in living costs between countries."
        #)
        #st.caption(
        #f"~ {(years_df['gdp_per_capita_worldbank'].mean() * (1 + gdp / 100)):.2f} int. $, PPP"
        #)  
        #st.space()  
          
        #Nurses & midwives per 1000
        nm = st.slider(
        "Increase :orange[*nurses/midwives per 1000 people*]",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
        format="%.1f%%",
        help="Nurses and midwives include professional nurses, professional midwives, auxiliary nurses & midwives, enrolled nurses & midwives and other associated personnel."
        )
        st.caption(
        f"~ {(years_df['nurses_and_midwives_per_1000_people'].mean() * (1 + nm / 100)):.2f} per 1000"
        )
        st.space()
        
        #Physicians per 1000
        phys = st.slider(
        "Increase :orange[*physicians per 1000 people*]",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,
        format="%.1f%%",
        help="Physicians include generalist and specialist medical practitioners."
        )
        st.caption(
        f"~ {(years_df['physicians_per_1000_people'].mean() * (1 + phys / 100)):.2f} per 1000"
        )
        st.space()
        
        #Prevalence of undernourishment
        undernourishment = st.slider(
        "Decrease :orange[*prevalence of undernourishment*]",
        #min_value=-100.0,
        min_value=-float(years_df['prevalence_of_undernourishment'].mean()),
        max_value=0.0,
        value=0.0,
        step=0.5,
        disabled=years_df['prevalence_of_undernourishment'].mean() <= 0.1,
        format="%.1f%%",
        help="Share of the population whose daily food intake does not provide enough energy to maintain a normal, active, and healthy life."
        )
        st.caption(
        f"~ {(years_df['prevalence_of_undernourishment'].mean() + undernourishment):.2f} %"
        )
        st.space()
        
        #Share of population urban 
        urban = st.slider(
        "Increase :orange[*share of population urban*]",
        min_value=0.0,
        max_value=100 - float(years_df["share_of_population_urban"].mean()),
        #max_value=float(100.0 / years_df["share_of_population_urban"].mean() -1) * 100, #50.0,
        value=0.0,
        step=0.5,
        format="%.1f%%",
        help="Share of the population living in urban areas."
        )
        st.caption(
        f"~ {( min(100.0, years_df['share_of_population_urban'].mean() + urban) ):.2f} %"
        )
        st.space()

        #Share without improved water 
        water = st.slider(
        "Decrease :orange[*share of population without improved water*]",
        min_value=-float(years_df['share_without_improved_water'].mean()),
        max_value=0.0,
        value=0.0,
        step=0.5,
        disabled=years_df['share_without_improved_water'].mean() <= 0.1,
        format="%.1f%%",
        help="Improved drinking water sources are those that have the potential to deliver safe water by nature of their design and construction, and include: piped water, boreholes or tubewells, protected dug wells, protected springs, rainwater, and packaged or delivered water."
        )
        st.caption(
        f"~ {( years_df['share_without_improved_water'].mean() + water ):.2f} %"
        )
        st.space()
        
        #vaccination coverage 
        vaccination = st.slider(
        "Increase :orange[*vaccination coverage*]",
        min_value=0.0,
        max_value=100 - float(years_df["vaccination_coverage_who_unicef"].mean()),
        value=0.0,
        step=0.5,
        format="%.1f%%",
        help="Share of one-year-olds who have had three doses of the combined diphtheria, tetanus and pertussis vaccine in a given year."
        )
        st.caption(
        f"~ {( min(100.0, years_df['vaccination_coverage_who_unicef'].mean() + vaccination) ):.2f} %"
        )
        st.space()
        
        #years of schooling 
        school = st.slider(
        "Increase :orange[*years of schooling*]",
        min_value=0.0,
        max_value=float(max(0.0, 14.0 -  years_df['years_of_schooling'].mean())),
        value=0.0,
        step=0.5,
        format="%.1f years",
        help="Average number of years women aged 25 and older have spent in formal education."
        )
        st.caption(
        f"~ {( years_df['years_of_schooling'].mean() + school):.1f} school years"
        )
        st.space()
        
        
        #modified_df["share_without_improved_water"] *= (1 + water / 100)
        modified_df = years_df.assign(
            annual_healthcare_expenditure_per_capita = lambda x: x["annual_healthcare_expenditure_per_capita"] * (1 + ahec / 100),
            #gdp_per_capita_worldbank = lambda x: x["gdp_per_capita_worldbank"] * (1 + gdp / 100),
            nurses_and_midwives_per_1000_people = lambda x: x["nurses_and_midwives_per_1000_people"] * (1 + nm / 100),
            physicians_per_1000_people = lambda x: x["physicians_per_1000_people"] * (1 + phys / 100),
            prevalence_of_undernourishment = lambda x: x["prevalence_of_undernourishment"] + undernourishment,
            share_of_population_urban = lambda x: x["share_of_population_urban"] + urban,
            share_without_improved_water = lambda x: x["share_without_improved_water"] + water,
            vaccination_coverage_who_unicef = lambda x: x["vaccination_coverage_who_unicef"] + vaccination,
            years_of_schooling = lambda x: x['years_of_schooling'] + school
        )  
        
        X = modified_df.drop(columns=["Entity", "Code", "Year", "child_mortality_igme"])
        predicts = modified_df.assign(
            pred_low  = qr_models["low"].predict(X),
            pred_med  = qr_models["med"].predict(X),
            pred_high = qr_models["high"].predict(X)
        )
        
        st.divider()
        if st.button("Simulate", type="primary"):
            st.session_state.show_msg = True


if years_select and country_select is not None:
    st.write("Original")
    years_df
    st.write("Modified")
    modified_df
# Show selected country
#st.write(country_selectbox)
#filtered_df = df[df["country"] == country_selectbox]
#st.dataframe(filtered_df)

if "show_msg" in st.session_state:
    st.info(f"(25%, 50%, 75%): "
        f"{predicts['pred_low'].mean():.2f}, {predicts['pred_med'].mean():.2f}, {predicts['pred_high'].mean():.2f}")
  
    #st.write("### Hello! This is appearing on the main page.")
    #st.balloons()
    
  
  

  
#modified_df['share_of_population_urban'].mean() * (1 + urban / 100)