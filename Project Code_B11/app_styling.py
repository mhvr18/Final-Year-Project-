import streamlit as st
import base64
import os

def load_custom_css():
    """
    Apply custom CSS styling to the entire application
    """
    custom_css = """
    <style>
        /* Improved text readability on image backgrounds */
        .stApp {
            color: #1E1E1E;
        }
        
        /* Card style for better content organization */
        .content-card {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        
        /* Custom buttons */
        .stButton>button {
            background-color: #3498DB;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 10px 15px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #2980B9;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
        }
        
        /* File uploader styling */
        .stFileUploader>div>div {
            background-color: rgba(255, 255, 255, 0.6);
            border-radius: 10px;
            padding: 10px;
        }
        
        /* Customize progress bar */
        .stProgress > div > div > div > div {
            background-color: #2ECC71;
        }
        
        /* Headers styling */
        h1, h2, h3 {
            color: #2C3E50;
            font-weight: bold;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }
        
        h2 {
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            font-size: 1.3rem;
            margin-bottom: 0.8rem;
        }
        
        /* Sidebar styling - additional elements */
        [data-testid="stSidebar"] .css-1d391kg {
            padding-top: 3.5rem;
        }
        
        /* Improve sidebar text contrast */
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: white !important;
            font-weight: 500;
        }
        
        /* Make selectbox and multiselect more visible */
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] .stMultiSelect > div > div {
            background-color: rgba(255, 255, 255, 0.9) !important;
            color: #1E1E1E !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def add_bg_from_local(image_file):
    """
    Add a background image to the main content area
    
    Parameters:
    -----------
    image_file : str
        Path to the image file
    """
    try:
        if os.path.exists(image_file):
            with open(image_file, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()
            
            background_style = f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """
            st.markdown(background_style, unsafe_allow_html=True)
        else:
            # If image doesn't exist, use a gradient background
            background_style = """
            <style>
            .stApp {
                background-color: #f5f7f9;
                background-image: linear-gradient(135deg, #f5f7f9 0%, #dfe9f3 100%);
            }
            </style>
            """
            st.markdown(background_style, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error setting background image: {str(e)}")

def sidebar_bg(image_file):
    """
    Add a background image to the sidebar
    
    Parameters:
    -----------
    image_file : str
        Path to the image file
    """
    try:
        if os.path.exists(image_file):
            file_ext = image_file.split('.')[-1]
            with open(image_file, "rb") as img_file:
                encoded_string = base64.b64encode(img_file.read()).decode()
            
            sidebar_style = f"""
            <style>
            [data-testid="stSidebar"] > div:first-child {{
                background-image: url("data:image/{file_ext};base64,{encoded_string}");
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
            }}
            </style>
            """
            st.markdown(sidebar_style, unsafe_allow_html=True)
        else:
            # If image doesn't exist, use a gradient background
            sidebar_style = """
            <style>
            [data-testid="stSidebar"] > div:first-child {
                background-image: linear-gradient(180deg, #2C3E50 0%, #4CA1AF 100%);
            }
            [data-testid="stSidebar"] .css-6qob1r.e1fqkh3o3 {
                color: white;
            }
            </style>
            """
            st.markdown(sidebar_style, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error setting sidebar background: {str(e)}")

def apply_card_style(content_func):
    """
    Decorator to wrap content in a styled card
    
    Parameters:
    -----------
    content_func : function
        Function that generates content
    
    Returns:
    --------
    wrapped_func : function
        Function that wraps the content in a styled card
    """
    def wrapper(*args, **kwargs):
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        result = content_func(*args, **kwargs)
        st.markdown('</div>', unsafe_allow_html=True)
        return result
    return wrapper

def apply_custom_styling():
    """
    Apply all custom styling elements to the application
    """
    # Apply base CSS styles
    load_custom_css()
    
    # Check for background images
    main_bg_options = ["background.jpg", "background.png", "bg.jpg", "bg.png"]
    sidebar_bg_options = ["sidebar_bg.jpg", "sidebar_bg.png", "sidebar.jpg", "sidebar.png"]
    
    # Try each potential background image file
    for bg_file in main_bg_options:
        if os.path.exists(bg_file):
            add_bg_from_local(bg_file)
            break
    
    # Try each potential sidebar image file
    for sb_file in sidebar_bg_options:
        if os.path.exists(sb_file):
            sidebar_bg(sb_file)
            break

# For testing the module
if __name__ == "__main__":
    st.set_page_config(page_title="Style Test", layout="wide")
    apply_custom_styling()
    
    st.title("Style Testing Page")
    st.write("This page demonstrates the custom styling of the application.")
    
    with st.container():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.subheader("This is a styled card")
        st.write("Content within the card is easier to read on image backgrounds.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.button("Test Button Styling")
    st.file_uploader("Test File Uploader Styling")