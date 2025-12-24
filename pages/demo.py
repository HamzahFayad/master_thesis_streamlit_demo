import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

st.title("Demo")


#reference dataset
df_ref = pd.read_csv("./reference_data/dataset.csv")


#sidebar
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

 
#percent_options = [0, 1, 2, 3, 5, 10, 20]
modified_df = years_df.copy()

with st.sidebar:
    if years_select and country_select is not None:
        st.divider()
        st.subheader(f"Adjust indicators to simulate hypothetical scenarios of child mortality rate in: :orange[*{country_select}*]")
        st.space()
        #Annual Healthcare Exp. per capita
        ahec = st.slider(
        "Increase annual health spending per capita",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        format="%.1f%%",
        help="Annual Healthcare expenditure per capita"
        )
        st.caption(
        f"{(years_df['annual_healthcare_expenditure_per_capita'].mean() * (1 + ahec / 100)):.2f} int. $"
        )
        st.space()
        #GDP per capita
        gdp = st.slider(
        "Increase gross domestic product per capita",
        min_value=0.0,
        max_value=50.0,
        value=0.0,
        step=5.0,
        format="%.1f%%"
        )
        st.caption(
        f"{(years_df['gdp_per_capita_worldbank'].mean() * (1 + gdp / 100)):.2f} int. $, PPP"
        )  
        st.space()    
        #Nurses & midwives per 1000
        nm = st.slider(
        "Increase nurses/midwives per 1000 people",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        format="%.1f%%"
        )
        st.caption(
        f"{(years_df['nurses_and_midwives_per_1000_people'].mean() * (1 + nm / 100)):.2f} per 1000"
        )
        st.space()
        #Physicians per 1000
        phys = st.slider(
        "Increase physicians per 1000 people",
        min_value=0.0,
        max_value=20.0,
        value=0.0,
        step=0.5,
        format="%.1f%%"
        )
        st.caption(
        f"{(years_df['physicians_per_1000_people'].mean() * (1 + phys / 100)):.2f} per 1000"
        )
        st.space()
        #Share without improved water 
        water = st.slider(
        "Decrease *share of population without improved water*",
        min_value=-50.0,
        max_value=0.0,
        value=0.0,
        step=0.5,
        format="%.1f%%"
        )
        st.caption(
        f"{(years_df['share_without_improved_water'].mean() * (1 + water / 100)):.2f} %"
        )
        
        
        st.divider()
        st.button("Simulate", type="primary")

        modified_df["share_without_improved_water"] *= (1 + water / 100)

if years_select and country_select is not None:
    st.write("Original")
    years_df
    st.write("Modified")
    modified_df
# Show selected country
#st.write(country_selectbox)
#filtered_df = df[df["country"] == country_selectbox]
#st.dataframe(filtered_df)
