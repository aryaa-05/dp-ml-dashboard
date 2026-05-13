# DP-ML Dashboard


## Project Overview
The **DP-ML Dashboard** is an interactive web application that demonstrates the application of Differential Privacy (DP) in Machine Learning. The project takes a survival analysis model trained on the METABRIC breast cancer dataset and provides a user-friendly UI to explore the dataset, view privacy parameters, analyze the privacy-utility tradeoff, and run live predictions.

## What is Differential Privacy?
Differential Privacy (DP) provides a mathematically rigorous guarantee that the inclusion or exclusion of a single individual in a dataset does not significantly affect the outcome of any analysis. In the context of Machine Learning, we use DP-SGD (Differentially Private Stochastic Gradient Descent) via libraries like `opacus`. It works by:
1. **Clipping gradients**: Limiting the influence of any single data point.
2. **Adding noise**: Injecting carefully calibrated Gaussian noise during training to obfuscate individual contributions.

This ensures the model learns general trends without memorizing sensitive patient records.

## Dataset Description
The dashboard uses the **METABRIC Breast Cancer Gene Expression Profiles** dataset.
- **Source**: Kaggle (`raghadalharbi/breast-cancer-gene-expression-profiles-metabric`)
- **Features**: Age at diagnosis, tumor size, mutation count, histologic grade, etc.
- **Target**: Overall survival (months & event status).

##  Methodology
1. **Data Preprocessing**: Handling missing values, scaling features, and removing redundant/highly correlated variables.
2. **Model Architecture**: A multi-layer perceptron (MLP) built with PyTorch for survival prediction.
3. **Privacy Engine**: Integrated `opacus.PrivacyEngine` to track the privacy budget (ε) and probability of failure (δ).
4. **Evaluation**: Computed Accuracy, Precision, Recall, F1-Score, and ROC-AUC. 
5. **Dashboard**: Built using `Streamlit` to visualize results dynamically.

## Results Summary
- **Privacy Budget (ε)**: 5.0
- **Accuracy**: ~85%
- **Tradeoff**: As privacy guarantees strengthen (lower ε), accuracy typically decreases. The dashboard includes interactive Plotly charts showing this exact relationship.

##  Screenshots
*(Add screenshots of your deployed dashboard here)*
- **Home Page**: `![Home](assets/Home.png)`
- **Tradeoff Chart**: `![Tradeoff](assets/Tradeoff.png)`

## 🚀 Installation & Local Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/dp-ml-dashboard.git
   cd dp-ml-dashboard
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment Instructions (Streamlit Community Cloud)

To deploy this app for free so anyone can access it:

1. **Push to GitHub** (see instructions below).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your `dp-ml-dashboard` repository.
5. Select the `main` branch.
6. Set the Main file path to `app.py`.
7. Click **"Deploy"**.

## 🐙 GitHub Setup Instructions

Run the following commands in your terminal from the project root to push this code to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: DP-ML dashboard"
git branch -M main
# Replace <your-username> with your actual GitHub username
git remote add origin https://github.com/<your-username>/dp-ml-dashboard.git
git push -u origin main
```

## 📁 Repository Structure

```text
dp-ml-dashboard/
├── notebook.ipynb         # Original Jupyter Notebook with DP training code
├── app.py                 # Main Streamlit application
├── src/                   # Python modules extracted from notebook
│   ├── __init__.py
│   ├── data_loader.py     # Kaggle data downloading & preprocessing
│   ├── model_utils.py     # PyTorch model definition & evaluation
│   ├── privacy_utils.py   # Opacus privacy engine configuration
│   └── visualization.py   # Plotly charting functions
├── results/               # Pre-computed metrics for the UI
│   ├── metrics.csv
│   ├── privacy_tradeoff.csv
│   ├── training_curves.csv
│   └── plots/             # Saved static plots
├── saved_models/          # Directory for exported model weights (.pth)
├── assets/                # Images, banners, and screenshots
├── requirements.txt       # Project dependencies
├── .gitignore             # Git ignore file
└── LICENSE                # MIT License
```

## 📺 Live Demo
*(Update this link after deploying to Streamlit Cloud)*
[View Live Dashboard](https://your-app-url.streamlit.app)
