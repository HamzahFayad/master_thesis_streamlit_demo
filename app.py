import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

title = ":orange[U5MR] - Child Mortality Rate"
about = '''
        This is a demo tool - part of a master thesis project titled:     
        *Use of machine learning to predict child mortality rates and identify relevant influencing indicators: A simulation-based country-level analysis.*
        
        Statistics show that childrens' survival chances are steadily increasing globally while differences between countries remain large.
        In 2022, around 5 million deaths of children under five were recorded (rate: 38 per 1000) - a 50% reduction since 2000.
        The SDG goal 3.2.1 intends to reduce the child mortality rate to 25 per 1000 worldwide until 2030.
        According to recent reports by UN IGME this goal will not be reached in over 60 countries as of now.
        
        This hypothetical simulation tool aims users to learn correlations between different 
        socioeconomic and health related indicators on child mortality rate per country.      
        Due to the highly sensitive topic and the use of only aggregated country-level data,
        this tool does not provide causal effects but rather just potential correlations between features and the target.
        It can provide first insights on which indicator has a big effect on the target value.
        
        All data come from *Our World in Data (https://ourworldindata.org/)*, primarly gathered from UN, WHO, World Bank, UNICEF, UN IGME. 
        '''
st.title(title)
st.markdown(about)

raw_df = pd.read_csv("./reference_data/raw_dataset.csv")
fig, ax = plt.subplots()
plt.figure(figsize=(15,2))
sns.lineplot(data=raw_df, x="Year", y="child_mortality_igme", ax=ax, errorbar=None)
ax.set_ylabel("U5MR per 1000 live births")
ax.set_title("Overall Global Trend of Child Mortality Rate")
st.pyplot(fig)


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
        st.subheader(f"Adjust indicators to simulate hypothetical scenarios of child mortality rate in *{country_select}*")
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
    modified_df
# Show selected country
#st.write(country_selectbox)
#filtered_df = df[df["country"] == country_selectbox]
#st.dataframe(filtered_df)
