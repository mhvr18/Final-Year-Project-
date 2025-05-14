import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime

# Set custom color theme
COLOR_THEME = ["#1ABC9C", "#3498DB", "#9B59B6", "#E74C3C", "#F1C40F", "#2ECC71"]

# Custom CSS for better UI
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
    
    /* Cards */
    .card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #2980B9, #6DD5FA);
        color: white;
        border-radius: 10px;
        padding: 20px 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
</style>
"""

# Helper function to parse salary
def parse_salary(salary_text):
    """Parse salary from text to numeric values"""
    if pd.isna(salary_text):
        return None, None
    
    # Extract numbers from the string
    numbers = re.findall(r'\d+[\d,]*\.?\d*', str(salary_text))
    if not numbers:
        return None, None
    
    # Convert to float, removing commas
    numbers = [float(num.replace(',', '')) for num in numbers]
    
    if len(numbers) >= 2:
        return min(numbers), max(numbers)
    elif len(numbers) == 1:
        return numbers[0], numbers[0]
    else:
        return None, None

@st.cache_data(ttl=3600)
def load_job_data():
    """Load and preprocess the job data"""
    try:
        # Load the cleaned job data
        df = pd.read_csv('jd_cleaned.csv')
        
        # Make sure required columns exist
        if 'SALARY_NUMERIC' not in df.columns:
            df['SALARY_NUMERIC'] = None
        
        # Create posting date (mock data for visualization)
        if 'POSTING_DATE' not in df.columns:
            # Create mock posting dates spread over the last 6 months
            import numpy as np
            from datetime import datetime, timedelta
            
            today = datetime.now()
            date_range = [today - timedelta(days=x) for x in range(180)]
            df['POSTING_DATE'] = np.random.choice(date_range, size=len(df))
            df['POSTING_DATE'] = pd.to_datetime(df['POSTING_DATE'])
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def display_metrics(df):
    """Display key metrics in columns"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{len(df):,}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Total Jobs</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        avg_salary = df['SALARY_NUMERIC'].mean() if 'SALARY_NUMERIC' in df.columns else None
        avg_salary_display = f"${avg_salary:,.0f}" if pd.notna(avg_salary) and avg_salary is not None else "N/A"
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{avg_salary_display}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Average Salary</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        top_role = df['ROLE'].value_counts().index[0] if 'ROLE' in df.columns and not df.empty else "N/A"
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{top_role}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Top Role</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        top_company = df['COMPANY'].value_counts().index[0] if 'COMPANY' in df.columns and not df.empty else "N/A"
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{top_company}</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Top Company</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

def extract_common_skills(df, top_n=20):
    """Extract most common skills from job descriptions"""
    if 'CLEANED DESCRIPTION' not in df.columns or df.empty:
        return pd.Series()
    
    # Common tech skills to look for
    common_skills = [
        'python', 'java', 'javascript', 'html', 'css', 'sql', 'react', 'angular', 'node', 
        'aws', 'azure', 'docker', 'kubernetes', 'machine learning', 'tensorflow', 'pytorch',
        'data analysis', 'data science', 'excel', 'tableau', 'power bi', 'figma', 'ui/ux',
        'agile', 'scrum', 'devops', 'ci/cd', 'git', 'github', 'rest api', 'microservices',
        'cloud computing', 'saas', 'cybersecurity', 'blockchain', 'artificial intelligence',
        'natural language processing', 'computer vision', 'deep learning', 'statistics',
        'project management', 'product management', 'marketing', 'seo', 'content writing'
    ]
    
    # Count occurrences of each skill
    skill_counts = {}
    for skill in common_skills:
        count = df['CLEANED DESCRIPTION'].str.contains(r'\b' + skill + r'\b', case=False).sum()
        if count > 0:
            skill_counts[skill] = count
    
    # Convert to Series and get top skills
    skills_series = pd.Series(skill_counts).sort_values(ascending=False).head(top_n)
    return skills_series

def create_salary_chart(df):
    """Create salary distribution chart"""
    if 'SALARY_NUMERIC' not in df.columns or df.empty:
        return None
    
    # Create salary bins
    salary_data = df.dropna(subset=['SALARY_NUMERIC'])
    if salary_data.empty:
        return None
    
    fig = px.histogram(
        salary_data, 
        x='SALARY_NUMERIC',
        nbins=20,
        title="Salary Distribution",
        labels={'SALARY_NUMERIC': 'Annual Salary ($)'},
        color_discrete_sequence=[COLOR_THEME[0]]
    )
    
    fig.update_layout(
        xaxis_tickprefix='$',
        xaxis_tickformat=',',
        bargap=0.1
    )
    
    return fig

def create_company_chart(df):
    """Create chart showing top companies by job count"""
    if 'COMPANY' not in df.columns or df.empty:
        return None
    
    # Get top companies
    top_companies = df['COMPANY'].value_counts().head(10)
    
    fig = px.bar(
        x=top_companies.index,
        y=top_companies.values,
        title="Top Companies Hiring",
        labels={'x': 'Company', 'y': 'Number of Jobs'},
        color_discrete_sequence=[COLOR_THEME[1]]
    )
    
    fig.update_layout(
        xaxis_title="Company",
        yaxis_title="Number of Jobs",
    )
    
    return fig

def create_position_chart(df):
    """Create chart showing top positions"""
    if 'POSITIONNAME' not in df.columns or df.empty:
        return None
    
    # Get top positions
    top_positions = df['POSITIONNAME'].value_counts().head(10)
    
    fig = px.bar(
        x=top_positions.values,
        y=top_positions.index,
        title="Top Job Positions",
        labels={'x': 'Number of Listings', 'y': 'Position'},
        color_discrete_sequence=[COLOR_THEME[2]],
        orientation='h'
    )
    
    fig.update_layout(
        xaxis_title="Number of Listings",
        yaxis_title="Position",
    )
    
    return fig

def create_skills_chart(df, top_n=15):
    """Create chart showing most in-demand skills"""
    skills = extract_common_skills(df, top_n)
    
    if skills.empty:
        return None
    
    fig = px.bar(
        x=skills.index,
        y=skills.values,
        title=f"Top {len(skills)} In-Demand Skills",
        labels={'x': 'Skill', 'y': 'Frequency in Job Descriptions'},
        color_discrete_sequence=[COLOR_THEME[3]]
    )
    
    fig.update_layout(
        xaxis_title="Skill",
        yaxis_title="Frequency",
        xaxis_tickangle=-45
    )
    
    return fig

def market_analysis_page():
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    st.title("📊 Job Market Analysis")
    st.write(
        "Explore current job market trends based on our database of job listings."
    )
    
    # Load job data
    with st.spinner("Loading job market data..."):
        df = load_job_data()
    
    if df.empty:
        st.error("No job data available. Please make sure the data cleaning process ran correctly.")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("Filter Data")
        
        # Company filter
        if 'COMPANY' in df.columns:
            top_companies = df['COMPANY'].value_counts().head(20).index.tolist()
            selected_companies = st.multiselect("Companies", options=top_companies)
        else:
            selected_companies = []
        
        # Role filter
        if 'ROLE' in df.columns:
            roles = sorted(df['ROLE'].unique())
            selected_roles = st.multiselect("Job Roles", options=roles)
        else:
            selected_roles = []
        
        # Salary range filter
        if 'SALARY_NUMERIC' in df.columns:
            min_salary = float(df['SALARY_NUMERIC'].min())
            max_salary = float(df['SALARY_NUMERIC'].max())
            
            salary_range = st.slider(
                "Salary Range ($)",
                min_value=min_salary,
                max_value=max_salary,
                value=(min_salary, max_salary),
                step=5000.0
            )
        else:
            salary_range = None
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_companies:
        filtered_df = filtered_df[filtered_df['COMPANY'].isin(selected_companies)]
    
    if selected_roles:
        filtered_df = filtered_df[filtered_df['ROLE'].isin(selected_roles)]
    
    if salary_range and 'SALARY_NUMERIC' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['SALARY_NUMERIC'] >= salary_range[0]) & 
            (filtered_df['SALARY_NUMERIC'] <= salary_range[1])
        ]
    
    # Display metrics
    display_metrics(filtered_df)
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Skills Demand", "Salary Insights", "Company Analysis"])
    
    with tab1:
        st.subheader("Skills in Demand")
        
        # Skills chart
        skills_chart = create_skills_chart(filtered_df)
        if skills_chart:
            st.plotly_chart(skills_chart, use_container_width=True)
        else:
            st.warning("Not enough data to analyze skills.")
        
        # Skills correlation heatmap (simple mock)
        if not filtered_df.empty and 'CLEANED DESCRIPTION' in filtered_df.columns:
            st.subheader("Skills often found together")
            
            # Create correlation matrix (mock data for visualization)
            skills = ["Python", "SQL", "Excel", "Machine Learning", "Data Analysis", "JavaScript", "React"]
            
            corr_matrix = [
                [1.0, 0.7, 0.5, 0.8, 0.7, 0.2, 0.1],
                [0.7, 1.0, 0.6, 0.5, 0.8, 0.3, 0.2],
                [0.5, 0.6, 1.0, 0.4, 0.7, 0.1, 0.1],
                [0.8, 0.5, 0.4, 1.0, 0.6, 0.1, 0.1],
                [0.7, 0.8, 0.7, 0.6, 1.0, 0.2, 0.1],
                [0.2, 0.3, 0.1, 0.1, 0.2, 1.0, 0.9],
                [0.1, 0.2, 0.1, 0.1, 0.1, 0.9, 1.0]
            ]
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=skills,
                y=skills,
                colorscale='Viridis',
                zmin=0, zmax=1
            ))
            
            fig.update_layout(
                title="Skills Correlation",
                xaxis_title="Skill",
                yaxis_title="Skill"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Salary Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary distribution
            salary_chart = create_salary_chart(filtered_df)
            if salary_chart:
                st.plotly_chart(salary_chart, use_container_width=True)
            else:
                st.warning("Not enough salary data for analysis.")
        
        with col2:
            # Salary by role
            if not filtered_df.empty and 'SALARY_NUMERIC' in filtered_df.columns and 'ROLE' in filtered_df.columns:
                salary_by_role = filtered_df.groupby('ROLE')['SALARY_NUMERIC'].mean().sort_values(ascending=False).head(10)
                
                fig = px.bar(
                    x=salary_by_role.index,
                    y=salary_by_role.values,
                    title="Average Salary by Role",
                    labels={'x': 'Role', 'y': 'Average Salary ($)'},
                    color_discrete_sequence=[COLOR_THEME[4]]
                )
                
                fig.update_layout(
                    xaxis_title="Role",
                    yaxis_title="Average Salary ($)",
                    yaxis_tickprefix='$',
                    yaxis_tickformat=',',
                    xaxis_tickangle=-45
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough data to analyze salary by role.")
    
    with tab3:
        st.subheader("Company Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top companies
            company_chart = create_company_chart(filtered_df)
            if company_chart:
                st.plotly_chart(company_chart, use_container_width=True)
            else:
                st.warning("Not enough company data for analysis.")
        
        with col2:
            # Top positions
            position_chart = create_position_chart(filtered_df)
            if position_chart:
                st.plotly_chart(position_chart, use_container_width=True)
            else:
                st.warning("Not enough position data for analysis.")
    
    # Data table view
    st.subheader("Raw Data")
    with st.expander("View Raw Data"):
        display_columns = [col for col in filtered_df.columns if col != 'CLEANED DESCRIPTION']
        st.dataframe(filtered_df[display_columns])
        
        # Download option
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv_data,
            file_name="job_market_data.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    market_analysis_page()