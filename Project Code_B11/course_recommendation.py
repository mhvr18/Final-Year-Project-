import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tempfile
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from skills_extraction import skills_extractor

# Initialize NLTK data
try:
    nltk.download("stopwords", quiet=True)
    stopw = set(stopwords.words("english"))
except:
    stopw = set()

# Set custom color theme
COLOR_THEME = ["#1ABC9C", "#3498DB", "#9B59B6", "#E74C3C", "#F1C40F", "#2ECC71"]

# Custom CSS for better UI
CSS_STYLES = """
<style>
    .course-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .course-card:hover {
        transform: translateY(-2px);
    }
    .match-score {
        color: #27AE60;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .skill-pill {
        background: #e9ecef;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
    }
    .platform-badge {
        background: #3498DB;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
</style>
"""

# Mock course data (in a real application, this would come from a database)
@st.cache_data
def load_course_data():
    # This is a mock dataset - in a real app, you'd fetch this from a database or API
    courses = [
        {
            "title": "Python for Data Science and Machine Learning",
            "platform": "Udemy",
            "instructor": "Jose Portilla",
            "rating": 4.7,
            "students": 418000,
            "duration": "25 hours",
            "level": "Intermediate",
            "skills": ["Python", "NumPy", "Pandas", "Matplotlib", "Seaborn", "Scikit-Learn", "Machine Learning", "Data Science"],
            "url": "https://www.udemy.com/course/python-for-data-science-and-machine-learning-bootcamp/",
            "description": "Learn how to use NumPy, Pandas, Seaborn, Matplotlib, Plotly, Scikit-Learn, Machine Learning, and more!"
        },
        {
            "title": "The Complete Web Developer Course",
            "platform": "Coursera",
            "instructor": "Dr. Angela Yu",
            "rating": 4.8,
            "students": 215000,
            "duration": "40 hours",
            "level": "Beginner",
            "skills": ["HTML", "CSS", "JavaScript", "Node.js", "React", "Web Development", "MongoDB", "Express.js"],
            "url": "https://www.coursera.org/specializations/web-development",
            "description": "Become a full-stack web developer with just one course. HTML, CSS, Javascript, Node, React, MongoDB, and more!"
        },
        {
            "title": "Machine Learning A-Z™: AI, Python & R",
            "platform": "Udemy",
            "instructor": "Kirill Eremenko",
            "rating": 4.5,
            "students": 692000,
            "duration": "44 hours",
            "level": "All Levels",
            "skills": ["Python", "R", "Machine Learning", "Data Science", "Deep Learning", "Statistics", "Data Analysis"],
            "url": "https://www.udemy.com/course/machinelearning/",
            "description": "Learn to create Machine Learning Algorithms in Python and R from two Data Science experts."
        },
        {
            "title": "AWS Certified Solutions Architect",
            "platform": "A Cloud Guru",
            "instructor": "Ryan Kroonenburg",
            "rating": 4.6,
            "students": 325000,
            "duration": "35 hours",
            "level": "Intermediate",
            "skills": ["AWS", "Cloud Computing", "S3", "EC2", "VPC", "Cloud Architecture", "IAM", "RDS"],
            "url": "https://acloudguru.com/course/aws-certified-solutions-architect-associate",
            "description": "Everything you need to pass the AWS Solutions Architect Associate exam and become certified."
        },
        {
            "title": "SQL - MySQL for Data Analytics and Business Intelligence",
            "platform": "Udemy",
            "instructor": "365 Careers",
            "rating": 4.7,
            "students": 186000,
            "duration": "10 hours",
            "level": "Beginner",
            "skills": ["SQL", "MySQL", "Database", "Business Intelligence", "Data Analytics", "Reporting", "Data Analysis"],
            "url": "https://www.udemy.com/course/sql-mysql-for-data-analytics-and-business-intelligence/",
            "description": "Learn how to use SQL for data analysis and business intelligence with this comprehensive course."
        },
        {
            "title": "JavaScript Algorithms and Data Structures",
            "platform": "freeCodeCamp",
            "instructor": "freeCodeCamp Team",
            "rating": 4.9,
            "students": 850000,
            "duration": "300 hours",
            "level": "Intermediate",
            "skills": ["JavaScript", "Algorithms", "Data Structures", "Programming", "Problem Solving", "Computational Thinking"],
            "url": "https://www.freecodecamp.org/learn/javascript-algorithms-and-data-structures/",
            "description": "Learn fundamental programming concepts in JavaScript including basic algorithms, object-oriented programming, and functional programming."
        },
        {
            "title": "Data Science Specialization",
            "platform": "Coursera",
            "instructor": "Johns Hopkins University",
            "rating": 4.6,
            "students": 540000,
            "duration": "180 hours",
            "level": "Intermediate",
            "skills": ["R", "Data Science", "Statistics", "Machine Learning", "Data Cleaning", "Data Visualization", "Regression Models"],
            "url": "https://www.coursera.org/specializations/jhu-data-science",
            "description": "Launch your career in data science. A ten-course introduction to data science, developed and taught by leading professors."
        },
        {
            "title": "TensorFlow Developer Certificate",
            "platform": "Coursera",
            "instructor": "DeepLearning.AI",
            "rating": 4.8,
            "students": 125000,
            "duration": "120 hours",
            "level": "Advanced",
            "skills": ["TensorFlow", "Deep Learning", "Neural Networks", "Computer Vision", "NLP", "CNN", "RNN", "Python"],
            "url": "https://www.coursera.org/professional-certificates/tensorflow-in-practice",
            "description": "Build and train neural networks using TensorFlow, improve network performance using convolutions, and apply ML techniques to complex scenarios."
        },
        {
            "title": "Excel Skills for Business Specialization",
            "platform": "Coursera",
            "instructor": "Macquarie University",
            "rating": 4.8,
            "students": 290000,
            "duration": "80 hours",
            "level": "Beginner to Advanced",
            "skills": ["Excel", "Data Analysis", "Business Analytics", "Pivot Tables", "Data Visualization", "VBA", "Macros"],
            "url": "https://www.coursera.org/specializations/excel",
            "description": "Master Excel tools and formulas to become an expert in data manipulation, analysis and visualization."
        },
        {
            "title": "The Complete Digital Marketing Course",
            "platform": "Udemy",
            "instructor": "Rob Percival",
            "rating": 4.5,
            "students": 320000,
            "duration": "22 hours",
            "level": "All Levels",
            "skills": ["Digital Marketing", "SEO", "Social Media Marketing", "Facebook Ads", "Google Analytics", "Content Marketing", "Email Marketing"],
            "url": "https://www.udemy.com/course/the-complete-digital-marketing-course/",
            "description": "Master Digital Marketing Strategy, Social Media Marketing, SEO, YouTube, Email, Facebook Marketing, Analytics & More!"
        },
        {
            "title": "The Complete React Developer Course",
            "platform": "Udemy",
            "instructor": "Andrew Mead",
            "rating": 4.7,
            "students": 182000,
            "duration": "39 hours",
            "level": "Intermediate",
            "skills": ["React", "JavaScript", "Redux", "Node.js", "Firebase", "Web Development", "Front-end Development"],
            "url": "https://www.udemy.com/course/react-2nd-edition/",
            "description": "Learn React by building real apps. Includes React Router, Next.js, Hooks, Redux, Firebase, and more."
        },
        {
            "title": "Google Data Analytics Professional Certificate",
            "platform": "Coursera",
            "instructor": "Google",
            "rating": 4.8,
            "students": 420000,
            "duration": "180 hours",
            "level": "Beginner",
            "skills": ["Data Analysis", "R", "SQL", "Tableau", "Excel", "Google Sheets", "Data Visualization", "Data Cleaning"],
            "url": "https://www.coursera.org/professional-certificates/google-data-analytics",
            "description": "Prepare for a new career in the high-growth field of data analytics, no experience required."
        }
    ]
    return pd.DataFrame(courses)

# Find courses matching user skills and needed skills
def recommend_courses(user_skills, needed_skills=None, preferences=None):
    """Recommend courses based on user skills and optional needed skills"""
    df = load_course_data()
    
    # Convert user skills to lowercase for matching
    user_skills_lower = [skill.lower() for skill in user_skills]
    
    # Create a list of all skills from the courses
    all_course_skills = [skill.lower() for course in df['skills'] for skill in course]
    unique_skills = sorted(list(set(all_course_skills)))
    
    # If needed skills not provided, recommend courses that build on existing skills
    if not needed_skills:
        needed_skills = []
        for skill in unique_skills:
            if skill not in user_skills_lower and any(us in skill or skill in us for us in user_skills_lower):
                needed_skills.append(skill)
    
    # Calculate match scores
    df['user_skill_match'] = df['skills'].apply(
        lambda x: sum(1 for skill in x if skill.lower() in user_skills_lower) / len(x) if x else 0
    )
    
    # Calculate needed skill match
    needed_skills_lower = [skill.lower() for skill in needed_skills]
    df['needed_skill_match'] = df['skills'].apply(
        lambda x: sum(1 for skill in x if skill.lower() in needed_skills_lower) / len(x) if x else 0
    )
    
    # Apply preferences if provided
    if preferences:
        if 'platform' in preferences and preferences['platform']:
            df = df[df['platform'].isin(preferences['platform'])]
        if 'level' in preferences and preferences['level']:
            df = df[df['level'].isin(preferences['level'])]
    
    # Final match score (weight needed skills higher)
    df['match_score'] = (df['user_skill_match'] * 0.4) + (df['needed_skill_match'] * 0.6)
    
    # Sort by match score
    return df.sort_values('match_score', ascending=False)

# Display a course card with formatting
def display_course_card(course):
    with st.container():
        st.markdown(f"<div class='course-card'>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{course['title']}**")
            st.markdown(f"<span class='platform-badge'>{course['platform']}</span> Instructor: {course['instructor']}", 
                        unsafe_allow_html=True)
            st.caption(f"⏱️ Duration: {course['duration']} | 🎓 Level: {course['level']} | ⭐ Rating: {course['rating']}/5 ({course['students']:,} students)")
        with col2:
            st.markdown(f"<div class='match-score'>{course['match_score']*100:.1f}% Match</div>", 
                        unsafe_allow_html=True)
        
        # Skills covered
        st.subheader("Skills you'll learn:")
        for skill in course['skills']:
            st.markdown(f"<span class='skill-pill'>{skill}</span>", unsafe_allow_html=True)
        
        # Course description
        st.markdown(f"**Description:** {course['description']}")
        
        # Link to course
        st.markdown(f"[View Course on {course['platform']}]({course['url']})")
        
        st.markdown("</div>", unsafe_allow_html=True)

def course_recommendation_page():
    st.markdown(CSS_STYLES, unsafe_allow_html=True)
    st.title("Course Recommendations")
    st.write(
        "Get personalized course recommendations to enhance your skills and increase job opportunities."
    )
    
    # Check if skills have been extracted from resume
    if 'extracted_skills' in st.session_state:
        user_skills = st.session_state['extracted_skills']
        st.success(f"We found these skills from your resume: {', '.join(user_skills)}")
    else:
        user_skills = []
    
    # Sidebar filters
    with st.sidebar:
        st.subheader("Customize Recommendations")
        
        # Allow user to edit skills or enter them manually
        skill_input = st.text_area(
            "Your Skills (comma separated)",
            value=", ".join(user_skills) if user_skills else "",
            help="Edit or add skills separated by commas"
        )
        
        # Parse skills from input
        if skill_input:
            user_skills = [skill.strip() for skill in skill_input.split(",") if skill.strip()]
        
        # Get skills you want to develop
        st.subheader("Skills to Develop")
        needed_skills_input = st.text_area(
            "Skills You Want to Learn (comma separated)",
            help="Enter skills you want to develop, separated by commas"
        )
        
        # Parse needed skills from input
        needed_skills = []
        if needed_skills_input:
            needed_skills = [skill.strip() for skill in needed_skills_input.split(",") if skill.strip()]
        
        # Platform filter
        platform_options = ["Udemy", "Coursera", "freeCodeCamp", "A Cloud Guru", "edX", "LinkedIn Learning"]
        selected_platforms = st.multiselect("Platforms", platform_options)
        
        # Level filter
        level_options = ["Beginner", "Intermediate", "Advanced", "All Levels"]
        selected_levels = st.multiselect("Learning Level", level_options)
        
        # Create preferences dict
        preferences = {
            'platform': selected_platforms,
            'level': selected_levels
        }
        
        num_courses = st.number_input("Number of courses to recommend", min_value=1, max_value=20, value=5)
    
    # Main section
    if not user_skills:
        st.warning("Please upload your resume in the Resume Analysis tab first or enter your skills manually.")
    
    if st.button("Get Course Recommendations", use_container_width=True):
        if user_skills:
            with st.spinner("Finding the best courses for your career growth..."):
                # Get course recommendations
                recommended_courses = recommend_courses(user_skills, needed_skills, preferences)
                
                # Display recommendations
                if recommended_courses.empty:
                    st.warning("No courses match your criteria. Try adjusting your filters.")
                else:
                    # Limit to requested number
                    recommended_courses = recommended_courses.head(num_courses)
                    
                    # Display insights
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Skills distribution
                        if not recommended_courses.empty:
                            all_skills = []
                            for skills_list in recommended_courses['skills']:
                                all_skills.extend(skills_list)
                            
                            skill_counts = pd.Series(all_skills).value_counts().head(10)
                            fig = px.bar(
                                x=skill_counts.index, 
                                y=skill_counts.values,
                                labels={'x': 'Skills', 'y': 'Frequency'},
                                title="Most Common Skills in Recommended Courses",
                                color_discrete_sequence=[COLOR_THEME[2]]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Platform distribution
                        platform_counts = recommended_courses['platform'].value_counts()
                        fig = px.pie(
                            values=platform_counts.values,
                            names=platform_counts.index,
                            title="Course Platforms",
                            color_discrete_sequence=COLOR_THEME
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display course cards
                    st.subheader(f"Top {len(recommended_courses)} Recommended Courses")
                    for _, course in recommended_courses.iterrows():
                        display_course_card(course)
        else:
            st.error("Please enter your skills before getting recommendations.")

if __name__ == "__main__":
    course_recommendation_page()