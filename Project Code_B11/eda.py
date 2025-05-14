import streamlit as st
import pandas as pd
import plotly.express as px
import re
from datetime import datetime
from db import fetch_data

# Constants
COLOR_THEME = ["#1ABC9C", "#3498DB", "#9B59B6", "#E74C3C", "#F1C40F", "#2ECC71"]
CSS_STYLES = """
<style>
    /* Main container */
    .main {
        background-color: #F8F9FA;
    }
    
    /* Titles */
    h1, h2, h3, h4, h5, h6 {
        color: #2C3E50;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(195deg, #2C3E50 0%, #3498DB 100%);
        color: white;
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
"""

@st.cache_data(ttl=3600)
def load_data():
    """Load and cache data with error handling"""
    try:
        df = fetch_data()
        if df is None or df.empty:
            return pd.DataFrame()
        df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def parse_salary(range_str):
    """Parse salary range into min and max values"""
    try:
        numbers = re.findall(r'£([\d,]+)', str(range_str))
        numbers = [float(num.replace(',', '')) for num in numbers]
        if len(numbers) == 2:
            return numbers[0], numbers[1]
        elif len(numbers) == 1:
            return numbers[0], numbers[0]
        return None, None
    except:
        return None, None

def apply_filters(df, locations, industries, experience, salary_range, date_range, search_query):
    """Apply all filters to the dataframe"""
    if search_query:
        df = df.loc[df['job_title'].str.contains(search_query, case=False, na=False)]
    
    if locations:
        df = df.loc[df['location'].isin(locations)]
    
    if industries:
        df = df.loc[df['industry'].isin(industries)]
    
    if experience:
        df = df.loc[df['experience_level'].isin(experience)]
    
    if salary_range:
        df = df.loc[(df['salary_min'] >= salary_range[0]) & (df['salary_max'] <= salary_range[1])]
    
    if date_range:
        start_date, end_date = date_range
        df = df.loc[(df['date_posted'] >= pd.Timestamp(start_date)) & (df['date_posted'] <= pd.Timestamp(end_date))]

    
    return df

def display_metrics(df):
    """Display key metrics in columns"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs", len(df))
    with col2:
        avg_salary = df['salary_max'].mean()
        st.metric("Avg Salary", f"£{avg_salary:,.0f}" if not pd.isna(avg_salary) else "N/A")
    with col3:
        top_industry = df['industry'].value_counts().idxmax() if not df.empty else "N/A"
        st.metric("Top Industry", top_industry)
    with col4:
        recent_post = df['date_posted'].max().strftime("%Y-%m-%d") if not df.empty else "N/A"
        st.metric("Latest Post", recent_post)

def create_skills_chart(df, top_n):
    """Create interactive skills chart"""
    skills = df['required_skills'].str.split(', ').explode().value_counts().head(top_n)
    fig = px.bar(skills, orientation='h', title=f"Top {top_n} In-Demand Skills", 
                 labels={'value': 'Job Count', 'index': 'Skill'}, 
                 color_discrete_sequence=[COLOR_THEME[0]])
    fig.update_layout(showlegend=False)
    return fig

def create_salary_distribution(df):
    """Create interactive salary distribution chart"""
    fig = px.violin(df, y='salary_max', box=True, title="Salary Distribution", 
                    labels={'salary_max': 'Annual Salary (£)'}, color_discrete_sequence=[COLOR_THEME[1]])
    fig.update_layout(yaxis_tickprefix='£', yaxis_tickformat=',.0f')
    return fig

def eda_page():
    # Apply custom styles
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    st.title("📊 Job Market Analysis Dashboard")
    
    # Load data with progress
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.warning("No data available. Please check your data source.")
        return
    
    # Preprocess data
    df[['salary_min', 'salary_max']] = df['salary_range'].apply(parse_salary).apply(pd.Series)
    
    # Sidebar Filters
    with st.sidebar:
        st.title("🔍 Filters")
        
        # Date Range Filter
        min_date = df['date_posted'].min()
        max_date = df['date_posted'].max()
        date_range = st.date_input("Select Date Range", (min_date.date(), max_date.date()), min_date.date(), max_date.date())

        # Convert date_range to datetime64[ns]
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1])

        
        # Salary Filter
        salary_min, salary_max = df['salary_min'].min(), df['salary_max'].max()
        salary_range = st.slider("Annual Salary (£)", int(salary_min), int(salary_max), (int(salary_min), int(salary_max)))
        
        # Text Search
        search_query = st.text_input("Search Job Titles")
        
        # Other Filters
        locations = st.multiselect("Locations", df['location'].unique())
        industries = st.multiselect("Industries", df['industry'].unique())
        experience = st.multiselect("Experience Levels", df['experience_level'].unique())
    
    # Apply filters
    filtered_df = apply_filters(df, locations, industries, experience, salary_range, date_range, search_query)
    
    if filtered_df.empty:
        st.warning("No jobs match the selected filters.")
        return
    
    # Display metrics
    display_metrics(filtered_df)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "💼 Job Details", "📥 Data Export"])
    
    with tab1:
        st.header("Market Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            top_n_skills = st.slider("Select number of skills", 3, 20, 10)
            st.plotly_chart(create_skills_chart(filtered_df, top_n_skills), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_salary_distribution(filtered_df), use_container_width=True)
    
    with tab2:
        st.header("Job Analysis")
        top_jobs = filtered_df.nlargest(10, 'salary_max')
        st.dataframe(top_jobs)
    
    with tab3:
        st.header("Export Data")
        csv = filtered_df.to_csv(index=False)
        st.download_button("Download CSV", csv, "jobs_data.csv", "text/csv")

if __name__ == "__main__":
    eda_page()
