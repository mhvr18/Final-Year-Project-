import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_job_data(input_file_path, output_file_path):
    """
    Clean and preprocess the job data from Excel file
    
    Parameters:
    -----------
    input_file_path : str
        Path to the input Excel file with job data
    output_file_path : str
        Path to save the cleaned CSV file
    """
    # Load the data
    print(f"Loading data from {input_file_path}...")
    df = pd.read_excel(input_file_path)
    
    # Drop unnecessary columns
    print("Dropping unnecessary columns...")
    columns_to_drop = [
        'jobType', 'descriptionHTML', 'externalApplyLink', 'id', 'isExpired', 
        'postedAt', 'postingDateParsed', 'scrapedAt', 'searchInput/country', 
        'searchInput/position', 'url', 'urlInput'
    ]
    
    # Only drop columns that exist
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_cols, inplace=True)
    
    # Filter for full-time jobs if the columns exist
    print("Filtering for full-time jobs...")
    job_type_cols = ['jobType/0', 'jobType/1', 'jobType/2', 'jobType/3']
    existing_job_type_cols = [col for col in job_type_cols if col in df.columns]
    
    if existing_job_type_cols:
        mask = df[existing_job_type_cols[0]].str.lower() == 'full-time'
        for col in existing_job_type_cols[1:]:
            mask = mask | (df[col].str.lower() == 'full-time')
        df = df[mask]
    
    # Drop job type columns after filtering
    print("Removing job type columns...")
    existing_job_type_cols = [col for col in job_type_cols if col in df.columns]
    if existing_job_type_cols:
        df.drop(columns=existing_job_type_cols, inplace=True)
    
    # Convert salary to numeric
    print("Converting salary to numeric values...")
    if 'salary' in df.columns:
        def convert_salary_to_numeric(salary):
            if pd.isna(salary):
                return np.nan
            # Remove non-numeric characters
            salary = re.sub(r'[^\d\.\-]', '', str(salary))
            
            if '-' in salary:
                # For salary ranges, take the average
                try:
                    parts = salary.split('-')
                    if len(parts) >= 2:
                        low, high = parts[0], parts[1]
                        return (float(low.strip()) + float(high.strip())) / 2
                except ValueError:
                    return np.nan
            else:
                try:
                    return float(salary)
                except ValueError:
                    return np.nan
        
        df['salary_numeric'] = df['salary'].apply(convert_salary_to_numeric)
    else:
        df['salary_numeric'] = np.nan
    
    # Select and rename columns
    print("Selecting and renaming columns...")
    columns = ['company', 'ROLE', 'positionName', 'description', 'salary_numeric', 'rating', 'reviewsCount']
    # Only select columns that exist
    existing_cols = [col for col in columns if col in df.columns]
    
    if not existing_cols:
        print("Warning: None of the expected columns found in the dataset.")
        df_new = df.copy()  # Use all columns if none of the expected ones exist
    else:
        df_new = df[existing_cols]
    
    # Uppercase all column names
    df_new.columns = df_new.columns.str.upper()
    
    # Download NLTK data if not already available
    print("Downloading NLTK resources...")
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"Warning: Could not download all NLTK resources: {str(e)}")
    
    # Initialize stopwords and lemmatizer
    try:
        stop_words = set(stopwords.words('english'))
    except:
        print("Warning: Could not load stopwords, using empty set instead.")
        stop_words = set()
    
    try:
        lemmatizer = WordNetLemmatizer()
    except Exception as e:
        print(f"Warning: Could not initialize lemmatizer: {str(e)}")
        # Define a simple passthrough lemmatizer as fallback
        class SimpleWordNetLemmatizer:
            def lemmatize(self, word, **kwargs):
                return word
        lemmatizer = SimpleWordNetLemmatizer()
    
    # Clean text function
    print("Cleaning job descriptions...")
    def clean_text(text):
        """
        Clean text by applying:
        - Lowercasing
        - Removing special characters and numbers
        - Tokenization
        - Stopword removal
        - Lemmatization
        """
        if not isinstance(text, str):
            return ''
        
        try:
            # Lowercasing
            text = text.lower()
            # Removing special characters and numbers
            text = re.sub(r'[^a-z\s]', '', text)
            # Tokenization
            words = word_tokenize(text)
            # Stopword removal and Lemmatization
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            # Joining back into a single string
            return ' '.join(words)
        except Exception as e:
            print(f"Warning: Error cleaning text: {str(e)}")
            # Return lowercase text as fallback
            return text.lower() if isinstance(text, str) else ''
    
    # Apply the cleaning function to job descriptions if column exists
    if 'DESCRIPTION' in df_new.columns:
        df_new['CLEANED DESCRIPTION'] = df_new['DESCRIPTION'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')
        
        # Drop the original description column
        df_new.drop(columns=['DESCRIPTION'], inplace=True)
    else:
        print("Warning: DESCRIPTION column not found, skipping text cleaning.")
    
    # Save the cleaned data
    print(f"Saving cleaned data to {output_file_path}...")
    df_new.to_csv(output_file_path, index=False)
    
    print(f"Data cleaning complete! Saved to {output_file_path}")
    print(f"Total jobs: {len(df_new)}")
    
    return df_new

def generate_data_insights(df):
    """
    Generate some basic insights from the cleaned data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned job data
    """
    print("\n--- Data Insights ---")
    
    # Number of jobs
    print(f"Total number of jobs: {len(df)}")
    
    # Average salary
    if 'SALARY_NUMERIC' in df.columns:
        avg_salary = df['SALARY_NUMERIC'].mean()
        if not pd.isna(avg_salary):
            print(f"Average salary: ${avg_salary:,.2f}")
        else:
            print("Average salary: Not available")
    
    # Top companies
    if 'COMPANY' in df.columns:
        top_companies = df['COMPANY'].value_counts().head(5)
        print("\nTop 5 companies:")
        for company, count in top_companies.items():
            print(f"- {company}: {count} jobs")
    
    # Top roles
    if 'ROLE' in df.columns:
        top_roles = df['ROLE'].value_counts().head(5)
        print("\nTop 5 roles:")
        for role, count in top_roles.items():
            print(f"- {role}: {count} jobs")

if __name__ == "__main__":
    input_path = "tech_JD.xlsx"
    output_path = "jd_cleaned.csv"
    
    try:
        df_cleaned = clean_job_data(input_path, output_path)
        generate_data_insights(df_cleaned)
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        print("Please ensure the Excel file is in the correct location and format.")