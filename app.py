import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# --- 1. Page Configuration ---
st.set_page_config(page_title="College Cutoff Predictor", page_icon="🎓", layout="centered")
st.title("🎓 College Cutoff Score Predictor")
st.write("Enter the admission details below to predict the expected cutoff score based on historical data.")

# --- 2. Load Data and Train Model (Cached for Speed) ---
@st.cache_resource
def load_and_train_model():
    # Load data
    df = pd.read_csv('cutoffs.csv')
    df = df[df['figure_kind'] == 'score'].dropna(subset=['figure'])
    
    # Fill missing categories
    categorical_cols = ['category', 'university', 'programme', 'round', 'college', 'seat_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            
    # Basic Outlier Removal to speed up frontend training
    Q1 = df['figure'].quantile(0.25)
    Q3 = df['figure'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['figure'] >= (Q1 - 1.5 * IQR)) & (df['figure'] <= (Q3 + 1.5 * IQR))]

    X = df[['year'] + categorical_cols]
    y = df['figure']

    # Build Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['year']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # Using fewer trees (n_estimators=50) so the web app loads quickly
        ('model', RandomForestRegressor(n_estimators=50, random_state=42)) 
    ])
    
    # Train model
    model.fit(X, y)
    return model, df

# Load the trained model and the dataframe (to populate dropdowns)
with st.spinner("Training the AI model... (This will only take a moment on startup)"):
    model, df = load_and_train_model()

# --- 3. User Input Interface ---
st.header("Applicant Details")

# Create two columns for a clean layout
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Admission Year", min_value=2024, max_value=2030, value=2024)
    university = st.selectbox("University", sorted(df['university'].unique()))
    programme = st.selectbox("Programme", sorted(df[df['university'] == university]['programme'].unique()))

with col2:
    category = st.selectbox("Category (e.g., UR, OBC)", sorted(df['category'].unique()))
    round_no = st.selectbox("Admission Round", sorted(df['round'].unique()))
    college = st.selectbox("Specific College", sorted(df['college'].unique()))

seat_type = st.selectbox("Seat Type", sorted(df['seat_type'].unique()))

# --- 4. Prediction Logic ---
if st.button("Predict Cutoff Score", type="primary"):
    # Create a dataframe from the user's input
    input_data = pd.DataFrame({
        'year': [year],
        'category': [category],
        'university': [university],
        'programme': [programme],
        'round': [round_no],
        'college': [college],
        'seat_type': [seat_type]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display Result
    st.success(f"### Predicted Cutoff Score: {prediction:.2f}")
    st.balloons()