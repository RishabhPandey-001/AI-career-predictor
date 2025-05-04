import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import difflib
import datetime
from pathlib import Path
import streamlit.components.v1 as components

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="AI Career Advisor", layout="wide")

# Configuration
DATA_FILE = "final_fixed_enriched_data.csv"
MODEL_FILE = "career_model.pkl"
FEEDBACK_FILE = "user_feedback.csv"
RETRAIN_INTERVAL = 7  # days

# Load data
def load_data():
    if Path(DATA_FILE).exists():
        try:
            df = pd.read_csv(DATA_FILE)
            df = df.replace('', np.nan)
            df['experience'] = df['experience'].fillna(0).astype(str)
            skill_cols = ['skill_1', 'skill_2', 'skill_3', 'skill_4']
            df['skills_combined'] = df[skill_cols].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
            df.columns = df.columns.str.strip()
        except Exception as e:
            st.error(f"Error loading data file: {str(e)}")
            df = pd.DataFrame()
    else:
        st.error("Data file not found")
        df = pd.DataFrame()

    if Path(FEEDBACK_FILE).exists():
        try:
            feedback_df = pd.read_csv(FEEDBACK_FILE)
        except:
            feedback_df = pd.DataFrame(columns=[
                'education_level', 'skills', 'experience', 'interested_technology', 
                'interested_career_area', 'type_of_company', 'predicted_job', 
                'actual_job', 'rating', 'comment', 'timestamp'
            ])
    else:
        feedback_df = pd.DataFrame(columns=[
            'education_level', 'skills', 'experience', 'interested_technology', 
            'interested_career_area', 'type_of_company', 'predicted_job', 
            'actual_job', 'rating', 'comment', 'timestamp'
        ])

    return df, feedback_df

# Save feedback
def save_feedback(feedback_df, feedback_data):
    feedback_data['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    updated_feedback = pd.concat([feedback_df, pd.DataFrame([feedback_data])], ignore_index=True)
    updated_feedback.to_csv(FEEDBACK_FILE, index=False)
    return updated_feedback

# Skill gap analysis
def analyze_skill_gap(user_skills, required_skills):
    if pd.isna(required_skills) or not required_skills:
        return [], {}

    user_skills = [s.strip().lower() for s in user_skills.split(',') if s.strip()]
    required_skills = [s.strip().lower() for s in required_skills.split(',') if s.strip()]

    missing_skills = []
    similar_skills = {}

    for req_skill in required_skills:
        if req_skill not in user_skills:
            matches = difflib.get_close_matches(req_skill, user_skills, n=1, cutoff=0.7)
            if matches:
                similar_skills[req_skill] = matches[0]
            else:
                missing_skills.append(req_skill)

    return missing_skills, similar_skills

# Train model
def train_model(df, retrain=False):
    if len(df) < 2:
        st.warning("Not enough data to train model (minimum 2 samples required)")
        return None

    required_cols = ['education_level', 'skills_combined', 'experience', 
                     'logical_quotient_rating', 'coding_skills_rating',
                     'self-learning_capability?', 'interested_technology', 
                     'interested_career_area', 'type_of_company', 
                     'management_or_technical', 'suggested_job_role']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        return None

    preprocessor = ColumnTransformer(
        transformers=[
            ('education', OneHotEncoder(handle_unknown='ignore'), ['education_level']),
            ('skills', TfidfVectorizer(), 'skills_combined'),
            ('experience', OneHotEncoder(handle_unknown='ignore'), ['experience']),
            ('logical_quotient', OneHotEncoder(handle_unknown='ignore'), ['logical_quotient_rating']),
            ('coding_skills', OneHotEncoder(handle_unknown='ignore'), ['coding_skills_rating']),
            ('learning_capability', OneHotEncoder(handle_unknown='ignore'), ['self-learning_capability?']),
            ('interest', OneHotEncoder(handle_unknown='ignore'), ['interested_technology']),
            ('company_type', OneHotEncoder(handle_unknown='ignore'), ['type_of_company']),
            ('role_type', OneHotEncoder(handle_unknown='ignore'), ['management_or_technical']),
            ('career_area', OneHotEncoder(handle_unknown='ignore'), ['interested_career_area'])
        ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ))
    ])

    try:
        X = df[['education_level', 'skills_combined', 'experience', 
                'logical_quotient_rating', 'coding_skills_rating',
                'self-learning_capability?', 'interested_technology', 
                'type_of_company', 'management_or_technical', 
                'interested_career_area']]
        y = df['suggested_job_role']
        model.fit(X, y)
        joblib.dump((model, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

# Main app
def main():
    df, feedback_df = load_data()
    if df.empty:
        st.error("No data available for training")
        return

    # Styling
    st.markdown("""
    <style>
        .stApp { background-color: #f9f9f9; }
        h1, h2, h3 { color: #003366; }
        .my-text { color: black; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎯 AI-Powered Career Prediction System")
    st.markdown("Unlock your ideal career path based on your skills, interests, and education.")

    with st.expander("📝 Fill Your Details", expanded=True):
        with st.form("career_form"):
            col1, col2 = st.columns(2)
            with col1:
                education = st.selectbox("🎓 Education Level", ["Bachelor", "Master", "PhD", "Diploma", "Other"])
                skill_1 = st.text_input("Skill 1")
                skill_2 = st.text_input("Skill 2")
                skill_3 = st.text_input("Skill 3")
                skill_4 = st.text_input("Skill 4")
                experience = st.selectbox("🧠 Experience Level", [str(i) for i in range(11)] + ["10+"])
                logical_quotient = st.slider("🧮 Logical Quotient Rating", 1, 10, 5)
                coding_skills = st.slider("💻 Coding Skills Rating", 1, 10, 5)
                self_learning = st.selectbox("📘 Self-Learning Capability", ["yes", "no"])
            with col2:
                interest = st.selectbox("💡 Interested Technology", [
                    "programming", "Management", "data engineering", "cloud computing",
                    "networks", "Software Engineering", "Computer Architecture", "IOT",
                    "parallel computing", "hacking", "Other"])
                career_area = st.selectbox("🎯 Career Interest Area", [
                    "testing", "system developer", "security", "developer",
                    "Business process analyst", "cloud computing", "Other"])
                company_type = st.selectbox("🏢 Preferred Company Type", [
                    "BPA", "Cloud Services", "product development", "Testing and Maintainance Services",
                    "SAaS services", "Web Services", "Finance", "Sales and Marketing",
                    "Product based", "Service Based", "Other"])
                role_type = st.selectbox("🧑‍💼 Role Type", ["Management", "Technical"])
                actual_job = st.text_input("(Optional) Your Dream Job Role")
            submitted = st.form_submit_button("🔍 Get Recommendation")

    if submitted:
        skills = [skill_1, skill_2, skill_3, skill_4]
        skills = [s for s in skills if s and str(s).strip() != 'nan']
        skills_str = ','.join(skills)

        input_data = {
            'education_level': education,
            'skills_combined': skills_str.lower(),
            'experience': experience,
            'logical_quotient_rating': str(logical_quotient),
            'coding_skills_rating': str(coding_skills),
            'self-learning_capability?': self_learning,
            'interested_technology': interest,
            'interested_career_area': career_area,
            'type_of_company': company_type,
            'management_or_technical': role_type
        }
        input_df = pd.DataFrame([input_data])

        model = train_model(df)
        if model is not None:
            try:
                job_role = model.predict(input_df)[0]
                st.success(f"## ✅ Recommended Job Role: {job_role}")

                with st.expander("📈 Skill Gap Analysis"):
                    st.markdown(f"**Your Skills:** {skills_str}")
                    similar_roles = df[df['suggested_job_role'] == job_role]
                    if not similar_roles.empty:
                        common_skills = similar_roles['skills_combined'].str.split(',').explode().value_counts()
                        top_skills = common_skills.head(5).index.tolist()
                        missing_skills = [skill for skill in top_skills if skill.lower() not in [s.lower() for s in skills]]
                        if missing_skills:
                            st.write("🔧 Skills You Might Consider Learning:")
                            st.write(', '.join(missing_skills))
                        else:
                            st.write("✅ Your skills align well with this role!")

                with st.expander("📘 Career Growth Tips"):
                    st.write(f"- Prefer companies in: **{company_type}**")
                    st.write(f"- Explore certifications in **{interest}** and **{career_area}**")
                    st.write("- Stay updated with the latest tech trends")

                with st.expander("💬 Feedback: Help Improve Our System"):
                    rating = st.slider("Rate this Recommendation", 1, 5, 3)
                    feedback_actual = st.text_input("Expected Job Role (optional)")
                    feedback_comment = st.text_area("Your Comments (optional)")
                    if st.button("📩 Submit Feedback"):
                        feedback_data = {
                            'education_level': education,
                            'skills': skills_str,
                            'experience': experience,
                            'interested_technology': interest,
                            'interested_career_area': career_area,
                            'type_of_company': company_type,
                            'predicted_job': job_role,
                            'actual_job': feedback_actual or actual_job or "",
                            'rating': rating,
                            'comment': feedback_comment
                        }
                        feedback_df = save_feedback(feedback_df, feedback_data)
                        st.success("✅ Thank you for your feedback!")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Please try again later.")

# Entry point
if __name__ == "__main__":
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        secrets_path.parent.mkdir(exist_ok=True)
        with open(secrets_path, "w") as f:
            f.write("admin_mode = false\n")
    main()
