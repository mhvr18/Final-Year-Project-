import streamlit as st

from app_styling import apply_custom_styling

st.set_page_config(
    page_title="Automated Resume Screening using AI-Powered Parsing and Job suggestion using K-NN with Cosine Similarity",
    layout="wide",
)

# Add logo at the top of the sidebar
try:
    st.sidebar.image("logo.jpg", width=150)
except:
    st.sidebar.title("Resume Analyzer")

st.sidebar.title("Navigation")

from job_recommendation import job_recommendation_page
from course_recommendation import course_recommendation_page
from market_analysis import market_analysis_page
page = st.sidebar.radio(
    "Go to",
    [
        "📄 Resume Analysis & Job Matching",
        "📚 Course Recommendations",
        "📊 Job Market Analysis",
    ],
)

if page == "📄 Resume Analysis & Job Matching":
    job_recommendation_page()
elif page == "📚 Course Recommendations":
    course_recommendation_page()
elif page == "📊 Job Market Analysis":
    market_analysis_page()

# st.set_page_config(page_title="Resume Analyzer & Job Recommender", layout="wide")

# Apply custom styling (background images and CSS)
apply_custom_styling()
