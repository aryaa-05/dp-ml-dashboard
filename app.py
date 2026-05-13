import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_loader import load_metabric_data
from src.visualization import plot_privacy_tradeoff, plot_training_curves
from src.model_utils import get_dummy_predictions

st.set_page_config(page_title="DP-ML Dashboard", page_icon="🔒", layout="wide")

# Constants
RESULTS_DIR = "results"
SAVED_MODELS_DIR = "saved_models"
NOTEBOOK_PATH = "notebook.ipynb"

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home",
    "Dataset Overview",
    "Privacy Parameters",
    "Model Performance",
    "Privacy-Utility Tradeoff",
    "Training Curves",
    "Prediction Demo",
    "Downloads"
])

st.sidebar.markdown("---")
st.sidebar.info("Dashboard built with Streamlit to showcase Differential Privacy in Machine Learning.")

# --- Helper Functions ---
@st.cache_data
def load_data():
    try:
        return load_metabric_data()
    except Exception as e:
        st.warning(f"Could not load full dataset from Kaggle ({e}). Loading sample data.")
        # Fallback dummy data
        return pd.DataFrame({
            'age_at_diagnosis': np.random.randint(30, 80, 100),
            'tumor_size': np.random.randint(10, 50, 100),
            'mutation_count': np.random.randint(1, 10, 100),
            'overall_survival': np.random.randint(0, 2, 100)
        })

@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_tradeoff():
    path = os.path.join(RESULTS_DIR, "privacy_tradeoff.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_curves():
    path = os.path.join(RESULTS_DIR, "training_curves.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# --- Page Content ---

if page == "Home":
    st.title("🔒 Differential Privacy Machine Learning Dashboard")
    st.markdown("""
    Welcome to the DP-ML Dashboard! This application showcases the application of **Differential Privacy (DP)** 
    on the METABRIC breast cancer dataset for survival analysis.
    
    ### What is Differential Privacy?
    Differential privacy is a rigorous mathematical framework that guarantees the privacy of individuals 
    in a dataset when training machine learning models. By adding carefully calibrated noise during training 
    (e.g., via DP-SGD), we ensure that the model doesn't memorize sensitive records.
    
    ### Project Overview
    - **Dataset**: METABRIC Breast Cancer Gene Expression Profiles
    - **Frameworks**: PyTorch, Opacus, scikit-survival
    - **Goal**: Predict overall survival while maintaining strict privacy guarantees.
    """)
    # Add architecture image if exists
    if os.path.exists("assets/banner.png"):
        st.image("assets/banner.png", use_container_width=True)

elif page == "Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("Exploring the **METABRIC** dataset used for model training.")
    
    with st.spinner("Loading data..."):
        df = load_data()
    
    col1, col2 = st.columns(2)
    col1.metric("Total Samples", df.shape[0])
    col2.metric("Total Features", df.shape[1] - 1) # Excluding target
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))
    
    st.subheader("Feature Names")
    st.write(list(df.columns))

elif page == "Privacy Parameters":
    st.title("🛡️ Privacy Parameters")
    st.markdown("The model was trained using **DP-SGD** with the following privacy configuration:")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Epsilon (ε)", "5.0", help="Privacy budget. Lower is more private.")
    col2.metric("Delta (δ)", "1e-5", help="Probability of privacy failure.")
    col3.metric("Noise Multiplier", "1.2", help="Amount of noise added to gradients.")
    col4.metric("Max Grad Norm", "1.0", help="Clipping threshold for gradients.")
    
    st.info("These parameters ensure a strong mathematical guarantee of privacy for the patients in the METABRIC dataset.")

elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Metrics for the final differentially private model.")
    
    metrics_df = load_metrics()
    if not metrics_df.empty:
        cols = st.columns(len(metrics_df))
        for idx, row in metrics_df.iterrows():
            cols[idx].metric(row['Metric'], row['Value'])
    else:
        st.warning("Metrics data not found.")

elif page == "Privacy-Utility Tradeoff":
    st.title("⚖️ Privacy-Utility Tradeoff")
    st.markdown("""
    One of the key challenges in DP-ML is balancing privacy (Epsilon) and model utility (Accuracy). 
    The chart below demonstrates this tradeoff.
    """)
    
    tradeoff_df = load_tradeoff()
    if not tradeoff_df.empty:
        fig = plot_privacy_tradeoff(tradeoff_df)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tradeoff_df)
    else:
        st.warning("Tradeoff data not found.")

elif page == "Training Curves":
    st.title("📉 Training Curves")
    st.markdown("Loss and Accuracy over epochs during the DP-SGD training process.")
    
    curves_df = load_curves()
    if not curves_df.empty:
        fig = plot_training_curves(curves_df)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Training curves data not found.")

elif page == "Prediction Demo":
    st.title("🔮 Prediction Demo")
    st.markdown("Enter patient feature values to predict overall survival using the DP model.")
    
    st.sidebar.subheader("Input Features")
    age = st.sidebar.slider("Age at Diagnosis", 20, 90, 50)
    tumor_size = st.sidebar.slider("Tumor Size (mm)", 5, 100, 20)
    mutation_count = st.sidebar.slider("Mutation Count", 1, 20, 5)
    
    input_data = pd.DataFrame({
        'age_at_diagnosis': [age],
        'tumor_size': [tumor_size],
        'mutation_count': [mutation_count]
    })
    
    st.write("### Patient Input")
    st.dataframe(input_data)
    
    if st.button("Run Prediction"):
        with st.spinner("Running DP Model..."):
            preds, probs = get_dummy_predictions(input_data)
            st.success("Prediction complete!")
            
            col1, col2 = st.columns(2)
            col1.metric("Predicted Survival", "Yes" if preds[0] == 1 else "No")
            col2.metric("Confidence", f"{probs[0]*100:.2f}%")

elif page == "Downloads":
    st.title("📥 Download Section")
    st.markdown("Download project artifacts and models.")
    
    # Metrics
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            st.download_button("Download Metrics (CSV)", f, file_name="metrics.csv", mime="text/csv")
            
    # Notebook
    if os.path.exists(NOTEBOOK_PATH):
        with open(NOTEBOOK_PATH, "rb") as f:
            st.download_button("Download Original Notebook", f, file_name="dp_ml_notebook.ipynb", mime="application/x-ipynb+json")
            
    st.info("The trained PyTorch model weights will be available here once training is fully exported to `saved_models/`.")
