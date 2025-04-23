import streamlit as st
import pandas as pd
import os
import base64
import openai
import subprocess
import sys
from streamlit_extras.add_vertical_space import add_vertical_space
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model, load_model, interpret_model, predict_model

# Ensure required packages are installed
try:
    import openai
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openai'])

# Set page configuration
st.set_page_config(
    page_title="AutoML Application",
    layout="wide",
    page_icon="🤖"
)

# ========================================
# ==========  Custom CSS/Style ===========
# ========================================
st.markdown(
    '''
    <style>
    /* ======== Chat History Hover => Purple (#8e44ad) ======== */
    .chat-history-item {
        padding: 6px;
        margin-bottom: 4px;
        border-radius: 4px;
        transition: background-color 0.1s ease-in-out;
    }
    .chat-history-item:hover {
        background-color: #8e44ad; 
        cursor: pointer;
    }
    button:hover, a:hover, [role="button"]:hover, label:hover, div[data-testid="stFileUploadDropzone"] div:hover {
        background-color: #8e44ad !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 10px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
        padding-bottom: 10px !important;
    }
    .stImage > img {
        border-radius: 50%;
        object-fit: cover;
    }
    .center-text {
        text-align: center;
    }
    .export-buttons {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-top: 10px;
    }
    </style>
    ''',
    unsafe_allow_html=True
)

# Display the logo in the sidebar
def display_logo():
    try:
        with open("logo.png", "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        st.sidebar.markdown(
            f'<div style="text-align:center;"><img src="data:image/png;base64,{img_base64}" width="200"></div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.sidebar.error(f"Error loading logo: {e}")

# Sidebar for navigation
with st.sidebar:
    display_logo()
    st.title("AutoML Application")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download", "Chatbot AI"])
    st.info("Build an automated ML pipeline using Streamlit, Pandas Profiling, and PyCaret.")
    st.write("Dr. Harry Patria.")

# Function to generate OpenAI response
def generate_openai_response(prompt, api_key, model="gpt-4"):
    """Generate response using OpenAI Chat API."""
    try:
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides detailed explanations specifically related to machine learning models, performance metrics, exploratory data analysis (EDA), data profiling, and feature importance."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {e}"

# Check for existing data
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

# Upload Section
if choice == "Upload":
    st.title("Upload Your Data for Modelling")
    file = st.file_uploader("Upload Your Dataset Here")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)

# Profiling Section
if choice == "Profiling" and 'df' in locals():
    st.title("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

# Modelling Section
if choice == "Modelling" and 'df' in locals():
    st.title("Model Training")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        setup(df, target=chosen_target)
        setup_df = pull()
        st.info("Experiment Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("Model Comparison")
        st.dataframe(compare_df)
        st.success("Best Model Selected")
        save_model(best_model, 'best_model')

# Chatbot AI Section
if choice == "Chatbot AI" and 'df' in locals():
    st.title("Ask the AI about Profiling, EDA, and Modelling")
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    user_question = st.text_area("Ask anything about models, performance metrics, EDA, data profiling, or feature importance")
    if st.button("Get Answer") and api_key and user_question:
        best_model = load_model('best_model')
        model_insights = interpret_model(best_model, plot='summary')
        prompt = f"Based on the uploaded data and the generated model, {user_question}. The feature importance insights are based on SHAP values: {model_insights}."
        response = generate_openai_response(prompt, api_key)
        st.write(response)

# Add footer
st.markdown(
    '''
    <div class="footer">
        <span class="disclaimer-icon" 
              title="Disclaimer: AI may not always provide accurate or complete information. 
                     Agentic AI x Corporate Learning Division">
              ℹ️
        </span>
        <span>All rights reserved. © 2025 Patria & Co.</span>
    </div>
    ''',
    unsafe_allow_html=True
)
