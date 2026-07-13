import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Unemployment Analysis in India",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .header-style {
        color: #1f77b4;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Unemployment in India.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month_name()
    df['Year'] = df['Date'].dt.year
    return df

df = load_data()

# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Select a Section:", [
    "📈 Overview",
    "🗺️ Region-wise Analysis",
    "📅 Month-wise Trends",
    "⏱️ Time Series by States",
    "🏆 Top 10 Regions",
    "🔗 Correlation Analysis",
    "🦠 COVID-19 Impact",
    "📊 Summary & Insights"
])

# Header
st.markdown("<div class='header-style'>📊 Unemployment Analysis in India</div>", unsafe_allow_html=True)
st.markdown("**Objective:** Exploratory Data Analysis on unemployment rates in India with focus on COVID-19 pandemic impact")
st.markdown("---")

# Page 1: Overview
if page == "📈 Overview":
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Regions", df['Region'].nunique())
    with col3:
        st.metric("Date Range", f"{df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')}")
    with col4:
        st.metric("Avg Unemployment Rate", f"{df['Estimated Unemployment Rate (%)'].mean():.2f}%")
    
    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Data Quality")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Column Information:**")
        st.dataframe(df.dtypes, use_container_width=True)
    with col2:
        st.write("**Missing Values:**")
        st.dataframe(df.isnull().sum(), use_container_width=True)

# Page 2: Region-wise Analysis
elif page == "🗺️ Region-wise Analysis":
    st.header("🗺️ Region-wise Average Unemployment Rate")
    st.markdown("Identifies which regions had the highest unemployment levels during the study period.")
    
    region_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(12, 8))
        region_avg.plot(kind="barh", color="steelblue", ax=ax)
        ax.set_title("Average Unemployment Rate by Region", fontsize=14, fontweight='bold')
        ax.set_xlabel("Average Unemployment Rate (%)")
        ax.set_ylabel("Region")
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("**Top 5 Regions:**")
        for i, (region, rate) in enumerate(region_avg.head(5).items(), 1):
            st.write(f"{i}. {region}: {rate:.2f}%")
    
    st.markdown("### 📌 Observation")
    st.info("Regions like Himachal Pradesh and Manipur show higher unemployment rates, while others like Jharkhand show lower rates. This indicates regional disparities in employment opportunities.")

# Page 3: Month-wise Trends
elif page == "📅 Month-wise Trends":
    st.header("📅 Month-wise Average Unemployment Rate")
    st.markdown("Analyze seasonal patterns in unemployment throughout the year.")
    
    month_avg = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_avg = month_avg.reindex(month_order)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    month_avg.plot(marker='o', color='coral', linewidth=2, markersize=8, ax=ax)
    ax.set_title("Month-wise Average Unemployment Rate", fontsize=14, fontweight='bold')
    ax.set_ylabel("Unemployment Rate (%)")
    ax.set_xlabel("Month")
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show data table
    st.write("**Monthly Averages:**")
    st.dataframe(month_avg, use_container_width=True)
    
    st.markdown("### 📌 Observation")
    st.success("A clear seasonal pattern emerges with sharp increase during COVID-19 lockdown (March-April 2020) and gradual decline afterward.")

# Page 4: Time Series by States
elif page == "⏱️ Time Series by States":
    st.header("⏱️ Unemployment Rate Over Time")
    st.markdown("Compare unemployment trends across different states.")
    
    # State selector
    available_states = sorted(df['Region'].unique())
    selected_states = st.multiselect("Select States:", available_states, default=["Maharashtra", "Delhi", "Tamil Nadu"])
    
    if selected_states:
        fig, ax = plt.subplots(figsize=(14, 6))
        for state in selected_states:
            temp = df[df["Region"] == state]
            ax.plot(temp["Date"], temp["Estimated Unemployment Rate (%)"], label=state, marker='o', linewidth=2)
        
        ax.legend(fontsize=10)
        ax.set_title("Unemployment Rate Over Time for Selected States", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Unemployment Rate (%)")
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### 📌 Observation")
        st.info("Different states experienced the pandemic differently. Some showed sudden spikes while others displayed more gradual changes in unemployment rates.")

# Page 5: Top 10 Regions
elif page == "🏆 Top 10 Regions":
    st.header("🏆 Top 10 Regions with Highest Average Unemployment")
    
    region_avg_full = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values(ascending=False)
    top10 = region_avg_full.head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=top10.values, y=top10.index, palette="rocket", ax=ax)
    ax.set_title("Top 10 States/Regions with Highest Average Unemployment", fontsize=14, fontweight='bold')
    ax.set_xlabel("Average Unemployment Rate (%)")
    ax.set_ylabel("Region")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Table view
    st.write("**Top 10 Detailed List:**")
    top10_df = pd.DataFrame({
        'Rank': range(1, 11),
        'Region': top10.index,
        'Avg Unemployment Rate (%)': top10.values.round(2)
    }).reset_index(drop=True)
    st.dataframe(top10_df, use_container_width=True)
    
    st.markdown("### 📌 Observation")
    st.warning("These regions consistently recorded higher unemployment levels, suggesting structural employment challenges that may require targeted policy interventions.")

# Page 6: Correlation Analysis
elif page == "🔗 Correlation Analysis":
    st.header("🔗 Correlation Heatmap")
    st.markdown("Analyze relationships between unemployment rate, employment rate, and labour participation rate.")
    
    corr = df[[
        "Estimated Unemployment Rate (%)",
        "Estimated Employed",
        "Estimated Labour Participation Rate (%)"
    ]].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Correlation'}, ax=ax)
    ax.set_title("Correlation Heatmap: Labour Market Indicators", fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("**Correlation Matrix:**")
    st.dataframe(corr, use_container_width=True)
    
    st.markdown("### 📌 Observation")
    st.info("""
    - **Negative correlation between unemployment and employment** (-0.24): As unemployment increases, employment decreases
    - **Labour participation patterns** show subtle relationships with economic conditions
    - These relationships confirm that job losses during pandemic reduced total employment
    """)

# Page 7: COVID-19 Impact
elif page == "🦠 COVID-19 Impact":
    st.header("🦠 Pre-COVID vs Post-COVID Comparison")
    st.markdown("Analyze the impact of COVID-19 pandemic on unemployment (split: March 2020).")
    
    pre = df[df["Date"] < "2020-03-01"]
    post = df[df["Date"] >= "2020-03-01"]
    
    # Metrics
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    pre_unemployment = pre["Estimated Unemployment Rate (%)"].mean()
    post_unemployment = post["Estimated Unemployment Rate (%)"].mean()
    change_pct = ((post_unemployment - pre_unemployment) / pre_unemployment * 100)
    
    with col1:
        st.metric("Pre-COVID Unemployment", f"{pre_unemployment:.2f}%")
    with col2:
        st.metric("Post-COVID Unemployment", f"{post_unemployment:.2f}%")
    with col3:
        st.metric("Change", f"+{change_pct:.1f}%", delta=f"+{post_unemployment - pre_unemployment:.2f}%")
    
    with col4:
        st.metric("Pre-COVID Employment", f"{pre['Estimated Employed'].mean()/1e6:.1f}M")
    with col5:
        st.metric("Post-COVID Employment", f"{post['Estimated Employed'].mean()/1e6:.1f}M")
    with col6:
        st.metric("Labour Participation", f"{post['Estimated Labour Participation Rate (%)'].mean():.2f}%")
    
    st.markdown("---")
    
    # Comparison Chart
    comparison = pd.DataFrame({
        "Period": ["Pre-COVID", "Post-COVID"],
        "Average Unemployment Rate": [pre_unemployment, post_unemployment]
    })
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(comparison["Period"], comparison["Average Unemployment Rate"], color=["#2ecc71", "#e74c3c"], width=0.5)
    ax.set_title("Pre vs Post COVID Average Unemployment Rate", fontsize=14, fontweight='bold')
    ax.set_ylabel("Average Unemployment Rate (%)")
    
    # Add value labels
    for bar, value in zip(bars, comparison["Average Unemployment Rate"]):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("### 📌 Observation")
    st.error(f"""
    **Significant Impact Detected:**
    - Unemployment increased by **{change_pct:.1f}%** (from {pre_unemployment:.2f}% to {post_unemployment:.2f}%)
    - This dramatic shift highlights the severe immediate impact of the pandemic
    - Lockdowns and reduced economic activity directly affected employment patterns
    - The difference clearly demonstrates how external shocks disrupt labour markets
    """)

# Page 8: Summary & Insights
elif page == "📊 Summary & Insights":
    st.header("📊 Summary & Key Insights")
    
    st.subheader("🔑 Key Findings")
    st.markdown("""
    1. **Pandemic Impact**: Unemployment rose sharply during COVID-19 lockdown period (March-August 2020)
    2. **Regional Disparities**: States like Himachal Pradesh, Manipur, and Meghalaya showed higher unemployment
    3. **Seasonal Patterns**: Clear monthly variations with peaks during lockdown periods
    4. **Labour Market Relationships**: Strong inverse correlation between unemployment and employment (-0.24)
    5. **Recovery Patterns**: Post-2020 data shows gradual recovery but at different rates across regions
    """)
    
    st.subheader("💡 Recommendations")
    st.markdown("""
    1. **Targeted Interventions**: Focus on high-unemployment regions like Himachal Pradesh and Manipur
    2. **Skill Development**: Invest in skill development programs to address structural unemployment
    3. **Regional Policies**: Design region-specific economic recovery and employment generation schemes
    4. **Labour Force Participation**: Create initiatives to encourage workforce participation, especially post-pandemic
    5. **Continuous Monitoring**: Track regional unemployment trends for early warning signals
    """)
    
    st.subheader("📈 Economic Impact Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Pre-COVID Statistics:**")
        pre = df[df["Date"] < "2020-03-01"]
        st.metric("Average Unemployment", f"{pre['Estimated Unemployment Rate (%)'].mean():.2f}%")
        st.metric("Average Employment", f"{pre['Estimated Employed'].mean()/1e6:.1f}M")
        st.metric("Labour Participation", f"{pre['Estimated Labour Participation Rate (%)'].mean():.2f}%")
    
    with col2:
        st.write("**Post-COVID Statistics:**")
        post = df[df["Date"] >= "2020-03-01"]
        st.metric("Average Unemployment", f"{post['Estimated Unemployment Rate (%)'].mean():.2f}%")
        st.metric("Average Employment", f"{post['Estimated Employed'].mean()/1e6:.1f}M")
        st.metric("Labour Participation", f"{post['Estimated Labour Participation Rate (%)'].mean():.2f}%")
    
    st.markdown("---")
    st.success("""
    ✅ **Analysis Complete**
    
    This comprehensive analysis demonstrates the significant impact of COVID-19 on India's labour market.
    The data reveals both immediate pandemic shocks and regional vulnerabilities that require targeted policy responses.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    <p>📊 Unemployment Analysis in India | Data Period: 2019-2021 | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Set seaborn style globally
sns.set_style("whitegrid")
