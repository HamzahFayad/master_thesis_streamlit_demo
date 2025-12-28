import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

st.set_page_config(layout="wide")

title = ":orange[U5MR] - Child Mortality Rate"
subtitle = ":orange[Simulate indicator effects on U5MR]"
about = '''
        This is a demo tool - part of a master thesis project titled:     
        *Use of machine learning to predict child mortality rates and identify relevant influencing indicators: A simulation-based country-level analysis.*
        
        Statistics show that childrens' survival chances are steadily increasing globally while differences between countries remain large.
        In 2022, around 5 million deaths of children under five were recorded (rate: 38 per 1000) - a 50% reduction since 2000.
        The SDG goal 3.2.1 intends to reduce the child mortality rate to 25 per 1000 worldwide until 2030.
        According to recent reports by UN IGME this goal will not be reached in over 60 countries as of now.
        
        This hypothetical simulation tool aims users to learn correlations between different 
        socioeconomic and health-related indicators on child mortality rate per country.
        In order for this simulation tool to work a regression-based supervised machine learning approach was applied on a country-level dataset to retrospectively predict under-five mortality rates.
        The final base dataset included 193 countries of 6 samples each from 2013 to 2018.
              
        Due to the highly sensitive topic and the use of only aggregated country-level data within a small period,
        this tool does not provide causal effects but rather just potential correlations between features and the target.
        It can provide first insights on which indicator has a big effect on the target value. With the dataset being highly heterogeneous, some extreme countries (e.g. South Sudan) get underpredicted.
        
        All data come from *Our World in Data (https://ourworldindata.org/)*, primarly gathered from UN, WHO, World Bank, UNICEF, UN IGME.
        Indicator *help descriptions* in the demos' sidebar were taken from Our World in Data. 
        '''
st.title(title)
st.subheader(subtitle)
st.markdown(about)
st.space()
st.page_link("pages/demo.py", label="Simulator Demo", icon="🌎")
st.divider()

@st.cache_data
def load_df():
    df = pd.read_csv("./reference_data/raw_dataset.csv")
    return df

raw_df = load_df()

col1, col2, col3 = st.columns([1, 4, 1])
with col2:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=raw_df, x="Year", y="child_mortality_igme", hue="world_regions_wb", ax=ax, errorbar=None)
        ax.set_ylabel("U5MR per 1000 live births")
        ax.set_title("Overall Global Trend of Child Mortality Rate")
        st.pyplot(fig)
    

st.divider()
st.caption("This project uses Machine Learning model to predict under-five mortality rates at country-level. For more check out the git repository: https://github.com/HamzahFayad/master_thesis_child_mortality")


#st.line_chart(raw_df, x="Year", y="child_mortality_igme", color="world_regions_wb", width="stretch")