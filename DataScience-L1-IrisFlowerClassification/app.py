import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Set page configuration for a premium, wide dashboard layout
st.set_page_config(
    page_title="Iris Species Classifier & Analyzer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        * {
            font-family: 'Outfit', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #7209B7 0%, #3F37C9 50%, #4895EF 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        
        .sub-header {
            text-align: center;
            color: #4A4E69;
            font-size: 1.2rem;
            margin-bottom: 2.5rem;
            font-weight: 300;
        }
        
        .card {
            background-color: #F8F9FA;
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #E9ECEF;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
            border-color: #DEE2E6;
        }
        
        .metric-title {
            color: #6C757D;
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .metric-value {
            color: #212529;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 5px;
        }
        
        .prediction-box {
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            color: white;
            font-weight: 700;
            font-size: 2.2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            margin-top: 15px;
        }
        
        .setosa-bg {
            background: linear-gradient(135deg, #4EA8DE 0%, #0077B6 100%);
        }
        
        .versicolor-bg {
            background: linear-gradient(135deg, #9D4EDD 0%, #7209B7 100%);
        }
        
        .virginica-bg {
            background: linear-gradient(135deg, #FF70A6 0%, #F72585 100%);
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #7209B7 0%, #3F37C9 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: linear-gradient(135deg, #8113CB 0%, #4E45E4 100%);
            box-shadow: 0 4px 15px rgba(114, 9, 183, 0.4);
            transform: translateY(-1px);
        }
    </style>
""", unsafe_allow_html=True)

# Cache data loading and model training
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["Species"] = iris.target
    df["Species_Name"] = df["Species"].map({
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    })
    return iris, df

@st.cache_resource
def train_models(X, y):
    lr = LogisticRegression(max_iter=200)
    lr.fit(X, y)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    
    return lr, knn

# Load raw materials
iris, df = load_data()
X = iris.data
y = iris.target
lr_model, knn_model = train_models(X, y)

# Header Section
st.markdown("<h1 class='main-header'>🌸 Iris Flower Classification Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>An interactive Data Science dashboard to explore the Iris dataset and make real-time ML predictions.</p>", unsafe_allow_html=True)

# Sidebar - User Inputs for Predictor
st.sidebar.markdown("### 🎛️ Input Physical Features")
st.sidebar.write("Adjust sliders to classify an offline flower sample:")

sepal_length = st.sidebar.slider("Sepal Length (cm)", 
                                float(df.iloc[:, 0].min()), 
                                float(df.iloc[:, 0].max()), 
                                float(df.iloc[:, 0].mean()), 
                                0.1)

sepal_width = st.sidebar.slider("Sepal Width (cm)", 
                               float(df.iloc[:, 1].min()), 
                               float(df.iloc[:, 1].max()), 
                               float(df.iloc[:, 1].mean()), 
                               0.1)

petal_length = st.sidebar.slider("Petal Length (cm)", 
                                float(df.iloc[:, 2].min()), 
                                float(df.iloc[:, 2].max()), 
                                float(df.iloc[:, 2].mean()), 
                                0.1)

petal_width = st.sidebar.slider("Petal Width (cm)", 
                               float(df.iloc[:, 3].min()), 
                               float(df.iloc[:, 3].max()), 
                               float(df.iloc[:, 3].mean()), 
                               0.1)

model_choice = st.sidebar.selectbox("🧠 Select Classifier Model", ["Logistic Regression", "K-Nearest Neighbors"])

# Main Dashboard Layout
tab_predict, tab_eda, tab_visualize = st.tabs(["🔮 Real-Time Predictor", "📊 Exploratory Data Analysis", "📈 Interactive Visualizations"])

with tab_predict:
    st.markdown("### 🌸 Real-Time Prediction Output")
    col_input, col_pred = st.columns([1, 1.2])
    
    with col_input:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### 📏 Current Input values")
        input_data = pd.DataFrame({
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Value (cm)": [sepal_length, sepal_width, petal_length, petal_width]
        })
        st.table(input_data)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_pred:
        # Prepare feature vector
        sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Classify
        if model_choice == "Logistic Regression":
            prediction = lr_model.predict(sample)[0]
            probabilities = lr_model.predict_proba(sample)[0]
        else:
            prediction = knn_model.predict(sample)[0]
            probabilities = knn_model.predict_proba(sample)[0]
            
        species_name = iris.target_names[prediction].title()
        
        # Color mapping
        bg_class = "setosa-bg" if prediction == 0 else ("versicolor-bg" if prediction == 1 else "virginica-bg")
        
        st.markdown(f"""
            <div class='prediction-box {bg_class}'>
                Classified Species: {species_name}
            </div>
        """, unsafe_allow_html=True)
        
        st.write("#### 📊 Prediction Probability Distribution:")
        for idx, prob in enumerate(probabilities):
            name = iris.target_names[idx].title()
            st.write(f"**{name}** ({prob*100:.1f}%)")
            st.progress(float(prob))

with tab_eda:
    st.markdown("### 📊 Dataset Overview")
    
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    with kpi_col1:
        st.markdown("""
            <div class='card'>
                <div class='metric-title'>Total Samples</div>
                <div class='metric-value'>150</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi_col2:
        st.markdown("""
            <div class='card'>
                <div class='metric-title'>Number of Features</div>
                <div class='metric-value'>4 Physical Features</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi_col3:
        st.markdown("""
            <div class='card'>
                <div class='metric-title'>Classes</div>
                <div class='metric-value'>Setosa, Versicolor, Virginica</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("#### 📑 Descriptive Statistics")
    st.dataframe(df.describe().T, use_container_width=True)
    
    st.write("#### 🔍 First 10 Samples Preview")
    st.dataframe(df.head(10), use_container_width=True)

with tab_visualize:
    st.markdown("### 📈 Custom Visualizations")
    
    col_ctrl, col_plot = st.columns([1, 2])
    
    with col_ctrl:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("#### Configure Visualization Parameters")
        x_axis = st.selectbox("Select X-Axis Feature", df.columns[:-2], index=2)
        y_axis = st.selectbox("Select Y-Axis Feature", df.columns[:-2], index=3)
        plot_type = st.radio("Select Plot Type", ["Scatter Plot", "3D Scatter Plot Plot"])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_plot:
        if plot_type == "Scatter Plot":
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color="Species_Name",
                title=f"{x_axis} vs {y_axis} Distributions",
                labels={"Species_Name": "Species"},
                color_discrete_map={"Setosa": "#4EA8DE", "Versicolor": "#7209B7", "Virginica": "#F72585"}
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family='Outfit'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            z_axis = st.selectbox("Select Z-Axis Feature for 3D Plot", df.columns[:-2], index=0)
            fig = px.scatter_3d(
                df,
                x=x_axis,
                y=y_axis,
                z=z_axis,
                color="Species_Name",
                title=f"3D Projection: {x_axis} vs {y_axis} vs {z_axis}",
                labels={"Species_Name": "Species"},
                color_discrete_map={"Setosa": "#4EA8DE", "Versicolor": "#7209B7", "Virginica": "#F72585"}
            )
            fig.update_layout(font_family='Outfit')
            st.plotly_chart(fig, use_container_width=True)

# Footer info
st.markdown("---")
st.markdown("<p style='text-align: center; color: #6C757D; font-size: 0.85rem;'>Developed using Python, Streamlit, and Plotly to showcase Machine Learning skills.</p>", unsafe_allow_html=True)
