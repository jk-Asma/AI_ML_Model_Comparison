# California House Price Prediction using Machine Learning
# Developed by Asma


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# Page Configuration

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}

.main{
    background-color:#F5F7FA;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

h1,h2,h3{
    color:#0F172A;
}

.hero{
    background:linear-gradient(135deg,#2563EB,#1D4ED8);
    padding:35px;
    border-radius:18px;
    color:white;
    box-shadow:0px 8px 18px rgba(0,0,0,0.18);
}

.card{
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 10px rgba(0,0,0,.10);
}

.metric-card{
    background:white;
    border-radius:15px;
    padding:20px;
    text-align:center;
    box-shadow:0px 3px 12px rgba(0,0,0,.12);
}

.prediction-card{
    background:linear-gradient(135deg,#16A34A,#22C55E);
    color:white;
    padding:30px;
    border-radius:18px;
    text-align:center;
    box-shadow:0px 6px 15px rgba(0,0,0,.20);
}

.footer{
    text-align:center;
    color:gray;
    margin-top:50px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# Project Directory
BASE_DIR = Path(__file__).parent

# Load Models
@st.cache_resource
def load_models():

    models = {
        "Decision Tree": joblib.load(BASE_DIR/"house_price_model.pkl"),
        "Linear Regression": joblib.load(BASE_DIR/"linear_regression.pkl"),
        "Ridge Regression": joblib.load(BASE_DIR/"ridge_regression.pkl")
    }

    return models

models = load_models()

best_model = models["Decision Tree"]


# Load Metrics

@st.cache_data
def load_metrics():

    df = pd.read_csv(
        BASE_DIR/"model_metrics.csv",
        index_col=0
    )

    df.reset_index(inplace=True)

    df.rename(
        columns={"index":"Model"},
        inplace=True
    )

    return df

metrics = load_metrics()


# Feature Names
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]


# Sidebar

st.sidebar.title(" California Housing")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        " House Price Prediction",
        " Model Analytics",
        " About Developer"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("""
### Best Model

 Decision Tree Regressor

Test R² : **0.6779**
""")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Developed by Asma\n\nAspiring AI/ML Engineer"
)


# Footer Function

def footer():

    st.markdown(
        """
<hr>

<div class='footer'>

<b>California House Price Prediction</b>

Developed with  using Streamlit & Scikit-Learn

<b>Asma</b>

Aspiring AI/ML Engineer | Intern at MainCraft Technology

</div>
""",
        unsafe_allow_html=True
    )


###### PAGE 1 : HOUSE PRICE PREDICTION


if page == " House Price Prediction":


    # Hero Section
    

    st.markdown("""
    <div class='hero'>
        <h1> California House Price Prediction</h1>
        <p style="font-size:18px;">
        Predict California house prices using a Machine Learning model trained on the
        California Housing Dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    left, right = st.columns([2,1])

   
    # Input Section
   
    with left:

        st.markdown("###  Enter Housing Details")

        c1, c2 = st.columns(2)

        with c1:

            medinc = st.number_input(
                "Median Income",
                min_value=0.0,
                max_value=20.0,
                value=3.50,
                step=0.10,
                help="Median income in the block."
            )

            averooms = st.number_input(
                "Average Rooms",
                min_value=1.0,
                max_value=20.0,
                value=5.50,
                step=0.10
            )

            population = st.number_input(
                "Population",
                min_value=1,
                value=1400,
                step=100
            )

            latitude = st.number_input(
                "Latitude",
                min_value=32.00,
                max_value=42.00,
                value=34.05,
                step=0.01
            )

        with c2:

            houseage = st.number_input(
                "House Age",
                min_value=1,
                max_value=60,
                value=28
            )

            avebedrms = st.number_input(
                "Average Bedrooms",
                min_value=0.50,
                max_value=10.00,
                value=1.10,
                step=0.10
            )

            aveoccup = st.number_input(
                "Average Occupancy",
                min_value=0.50,
                max_value=20.00,
                value=3.00,
                step=0.10
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-125.00,
                max_value=-114.00,
                value=-118.25,
                step=0.01
            )

   
    # Information Card
   
    with right:

        st.markdown("""
        <div class='card'>
        <h3>ℹ️ Prediction Model</h3>

        <b>Dataset</b><br>
        California Housing Dataset

        <br><br>

        <b>Algorithm</b><br>
        Decision Tree Regressor

        <br><br>

        <b>Features Used</b>

        <ul>
        <li>Median Income</li>
        <li>House Age</li>
        <li>Average Rooms</li>
        <li>Average Bedrooms</li>
        <li>Population</li>
        <li>Average Occupancy</li>
        <li>Latitude</li>
        <li>Longitude</li>
        </ul>

        </div>
        """, unsafe_allow_html=True)

    st.write("")

    
    # Prediction Button


    if st.button(" Predict House Price", use_container_width=True):

        try:

            input_data = np.array([[
                medinc,
                houseage,
                averooms,
                avebedrms,
                population,
                aveoccup,
                latitude,
                longitude
            ]])

            with st.spinner("Predicting..."):

                prediction = best_model.predict(input_data)[0]

            predicted_price = prediction * 100000

            st.write("")

            st.markdown(f"""
            <div class='prediction-card'>

            <h2>Estimated House Price</h2>

            <h1>${predicted_price:,.2f}</h1>

            <p>Prediction generated using the <b>Decision Tree Regressor</b>.</p>

            </div>
            """, unsafe_allow_html=True)

            st.write("")

            st.subheader(" Input Summary")

            s1, s2, s3, s4 = st.columns(4)

            s1.metric("Median Income", f"{medinc:.2f}")
            s2.metric("House Age", f"{houseage} yrs")
            s3.metric("Population", f"{population:,}")
            s4.metric("Average Occupancy", f"{aveoccup:.2f}")

        except Exception as e:

            st.error(f"Prediction failed: {e}")

    footer()


##### PAGE 2 : MODEL ANALYTICS


elif page == " Model Analytics":


    # Header
    
    st.markdown("""
    <div class='hero'>
        <h1>📊 Model Performance Analytics</h1>
        <p style="font-size:18px;">
        Comparative analysis of the three Machine Learning models trained
        on the California Housing Dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    df = metrics.copy()

    
    # Best Model
   

    best = df.loc[df["Test R2"].idxmax()]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            " Best Model",
            best["Model"]
        )

    with c2:
        st.metric(
            "Highest Test R²",
            f"{best['Test R2']:.4f}"
        )

    with c3:
        st.metric(
            "Lowest Test RMSE",
            f"{df['Test RMSE'].min():.4f}"
        )

    st.divider()

    
    # Test R2
   

    st.subheader(" Test R² Score Comparison")

    fig1 = px.bar(
        df,
        x="Model",
        y="Test R2",
        color="Model",
        text="Test R2",
        template="plotly_white"
    )

    fig1.update_traces(
        texttemplate="%{text:.4f}",
        textposition="outside"
    )

    fig1.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="Regression Models",
        yaxis_title="Test R² Score"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    
    # RMSE
    

    st.subheader(" Test RMSE Comparison")

    fig2 = px.bar(
        df,
        x="Model",
        y="Test RMSE",
        color="Model",
        text="Test RMSE",
        template="plotly_white"
    )

    fig2.update_traces(
        textposition="outside"
    )

    fig2.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="Regression Models",
        yaxis_title="RMSE"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    
    # Train vs Test R2
    

    st.subheader(" Train vs Test R²")

    fig3 = go.Figure()

    fig3.add_trace(
        go.Bar(
            name="Train R²",
            x=df["Model"],
            y=df["Train R2"]
        )
    )

    fig3.add_trace(
        go.Bar(
            name="Test R²",
            x=df["Model"],
            y=df["Test R2"]
        )
    )

    fig3.update_layout(
        barmode="group",
        template="plotly_white",
        height=450,
        xaxis_title="Regression Models",
        yaxis_title="R² Score"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.divider()

   
    # Overfitting Gap
   

    st.subheader(" Overfitting Gap Comparison")

    fig4 = px.line(
        df,
        x="Model",
        y="Overfit Gap (R2)",
        markers=True,
        template="plotly_white"
    )

    fig4.update_traces(
        line=dict(width=4),
        marker=dict(size=10)
    )

    fig4.update_layout(
        height=450,
        xaxis_title="Regression Models",
        yaxis_title="Overfit Gap"
    )

    st.plotly_chart(fig4, use_container_width=True)

    st.divider()

    
    # Insights
   

    st.subheader(" Model Performance Summary")

    st.success(f"""
 **Best Model:** {best['Model']}

 Highest Test R² Score: **{best['Test R2']:.4f}**

 Lowest Test RMSE: **{best['Test RMSE']:.4f}**

This model was selected for deployment in the prediction system.
""")

    st.info("""
### Key Insights

• **Linear Regression** provides a simple baseline with consistent performance.

• **Ridge Regression** offers similar performance while reducing model complexity through L2 regularization.

• **Decision Tree Regressor** achieved the highest predictive accuracy (**Test R² = 0.6779**) and the lowest prediction error (**RMSE = 0.6497**), making it the most suitable model for deployment.
""")

    footer()


#### PAGE 3 : ABOUT DEVELOPER


elif page == " About Developer":

    st.markdown("""
    <div class='hero'>
        <h1> About the Developer</h1>
        <p style="font-size:18px;">
        Passionate about Artificial Intelligence, Machine Learning, and Data Science.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    col1, col2 = st.columns([1,2])

    # --------------------------------------------------------
    # Left Profile Card
    # --------------------------------------------------------

    with col1:

        st.markdown("""
        <div class='card'>

        <h2 style="text-align:center;"> Asma</h2>

        <hr>

        <p><b>Role</b><br>
        Aspiring AI/ML Engineer</p>

        <p><b>Current Position</b><br>
        Intern at MainCraft Technology</p>

        <p><b>Project</b><br>
        California House Price Prediction</p>

        <p><b>Domain</b><br>
        Machine Learning</p>

        </div>
        """, unsafe_allow_html=True)

    
    # Right Content
  

    with col2:

        st.markdown("##  About Me")

        st.write("""
I am **Asma**, an aspiring **Artificial Intelligence and Machine Learning Engineer**
with a passion for solving real-world problems using data-driven solutions.

This project demonstrates my ability to build, evaluate, compare, and deploy
Machine Learning models through an interactive Streamlit application.
""")

        st.markdown("---")

        st.markdown("##  Project Overview")

        st.write("""
The **California House Price Prediction System** predicts housing prices using
Machine Learning techniques and compares multiple regression models to identify
the most suitable model for deployment.

The deployed prediction model is the **Decision Tree Regressor**, selected based
on its superior predictive performance.
""")

        st.markdown("---")

        st.markdown("##  Technologies Used")

        tech1, tech2 = st.columns(2)

        with tech1:

            st.success(" Python")
            st.success(" Pandas")
            st.success(" NumPy")
            st.success(" Scikit-Learn")

        with tech2:

            st.success(" Streamlit")
            st.success(" Plotly")
            st.success(" Joblib")
            st.success(" Git & GitHub")

        st.markdown("---")

        st.markdown("##  Machine Learning Models")

        st.info("""
✔ Linear Regression

✔ Ridge Regression

✔ Decision Tree Regressor (Best Model)
""")

    st.write("")

    # --------------------------------------------------------
    # Workflow
    # --------------------------------------------------------

    st.subheader(" Project Workflow")

    workflow = [
        " Data Collection",
        " Data Preprocessing",
        " Exploratory Data Analysis",
        " Model Training",
        " Model Evaluation",
        " Best Model Selection",
        " Streamlit Deployment"
    ]

    for step in workflow:
        st.write(step)

    st.markdown("---")

    
    # Skills

    st.subheader(" Technical Skills")

    skills = [
        "Python",
        "Machine Learning",
        "Data Analysis",
        "Data Visualization",
        "Model Evaluation",
        "Streamlit",
        "Scikit-Learn",
        "Git & GitHub"
    ]

    cols = st.columns(4)

    for i, skill in enumerate(skills):
        cols[i % 4].success(skill)

    st.markdown("---")

   
    # Contact
    st.subheader(" Contact")

    st.write("✉️ **Email:** asma0598amu@gmail.com")
    st.write("🔗 **LinkedIn:** www.linkedin.com/in/asma0598")
    st.write("💻 **GitHub:** https://github.com/jk-Asma")

    footer()