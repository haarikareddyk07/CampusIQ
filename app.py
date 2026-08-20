import os
import streamlit as st
import pandas as pd
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv

# Load API keys
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="CampusIQ - AI Opportunity Agent", layout="wide")

# App Header
st.title("🎓 CampusIQ: AI Opportunity Agent")
st.caption("Find tailored hackathons, internships & auto-generate application assets.")

# Mock Dataset of Opportunities
OPPORTUNITIES = [
    {
        "Title": "Smart India Hackathon 2026",
        "Type": "Hackathon",
        "Deadline": "2026-09-15",
        "Domain": "Web Dev, AI, IoT",
        "Description": "National level hackathon to solve real-world government & industry problems."
    },
    {
        "Title": "Google Summer of Code",
        "Type": "Open Source Internship",
        "Deadline": "2026-10-01",
        "Domain": "Python, Machine Learning, Open Source",
        "Description": "Global online program focused on bringing new contributors into open source software development."
    },
    {
        "Title": "Microsoft Engage '26",
        "Type": "Mentorship / Internship",
        "Deadline": "2026-09-05",
        "Domain": "Data Science, Web Development",
        "Description": "Mentorship program designed to help students learn engineering skills through real-world projects."
    }
]

# Sidebar: Resume Upload
st.sidebar.header("📄 Student Profile")
uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF)", type=["pdf"])

resume_text = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        resume_text += page.extract_text() or ""
    st.sidebar.success("Resume Parsed Successfully!")

# Display Opportunities Table
st.subheader("🚀 Active Opportunities")
df = pd.DataFrame(OPPORTUNITIES)
st.dataframe(df, width="stretch")

# Feature: AI Matcher & Copilot
st.divider()
st.subheader("⚡ AI Application Copilot")

selected_opp = st.selectbox("Select an opportunity to apply for:", [op["Title"] for op in OPPORTUNITIES])
target_data = next(item for item in OPPORTUNITIES if item["Title"] == selected_opp)

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Calculate Resume Match Score"):
        if not resume_text:
            st.warning("Please upload a resume in the sidebar first!")
        else:
            with st.spinner("Analyzing match..."):
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                prompt = f"""
                Analyze this student's resume against the opportunity details.
                Resume: {resume_text}
                Opportunity: {target_data['Title']} - {target_data['Description']} (Domains: {target_data['Domain']})

                Provide:
                1. Match Percentage (0-100%)
                2. Key Strengths
                3. Missing Skills/Gaps
                """
                response = model.generate_content(prompt)
                st.write(response.text)

with col2:
    if st.button("✍️ Generate Custom SOP / Essay"):
        if not resume_text:
            st.warning("Please upload a resume in the sidebar first!")
        else:
            with st.spinner("Drafting custom application..."):
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                prompt = f"""
                Write a tailored 200-word Statement of Purpose/Application Essay for this student applying to {target_data['Title']}.
                Student Resume Context: {resume_text}
                Opportunity Details: {target_data['Description']}
                """
                response = model.generate_content(prompt)
                st.text_area("Generated Output", response.text, height=250)
