import os
import time
import re
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

# Page setup
st.set_page_config(page_title="CampusIQ - AI Opportunity Agent", layout="wide")

# Load API keys & initialize Gemini Client
load_dotenv()

api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is missing! Please configure it in your Streamlit secrets or environment variables.")
    st.stop()

client = genai.Client(api_key=api_key)

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
        "Description": "Mentorship program designed to help students learn engineering skills through real-world problems."
    }
]

# Sidebar: Resume Upload
st.sidebar.header("📁 Student Profile")
uploaded_file = st.sidebar.file_uploader("Upload your Resume (PDF)", type=["pdf"])

resume_text = ""
if uploaded_file:
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            resume_text += extracted
    st.sidebar.success("Resume Parsed Successfully!")

# Display Opportunities Table
st.subheader("📌 Active Opportunities")
df = pd.DataFrame(OPPORTUNITIES)
st.dataframe(df, use_container_width=True)

# Feature: AI Matcher & Copilot
st.divider()
st.subheader("⚡ AI Application Copilot")

selected_opp = st.selectbox("Select an opportunity to apply for:", [op["Title"] for op in OPPORTUNITIES])
target_data = next(item for item in OPPORTUNITIES if item["Title"] == selected_opp)


def generate_content_with_retry(client, prompt, retries=3, delay=2):
    """Helper function to retry model requests on 503 high-demand errors."""
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            raise e


col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Calculate Resume Match Score"):
        if not resume_text:
            st.warning("Please upload a resume in the sidebar first!")
        else:
            with st.spinner("Analyzing match..."):
                prompt = f"""
Analyze this student's resume against the opportunity details.
Resume: {resume_text}
Opportunity: {target_data['Title']} - {target_data['Description']} (Domains: {target_data['Domain']})

Provide output strictly in this format:
Match Percentage: XX%
Key Strengths:
- ...
Missing Skills/Gaps:
- ...
"""
                try:
                    response = generate_content_with_retry(client, prompt)
                    text_output = response.text
                    
                    # Extract percentage for visual display
                    match_score = re.search(r"(\d{1,3})%", text_output)
                    if match_score:
                        score_val = int(match_score.group(1))
                        st.metric(label="Match Score", value=f"{score_val}%")
                        st.progress(score_val / 100)
                    
                    st.markdown(text_output)
                except Exception as e:
                    st.error(f"Failed to calculate match score: {e}")

with col2:
    if st.button("✍️ Generate Custom SOP / Essay"):
        if not resume_text:
            st.warning("Please upload a resume in the sidebar first!")
        else:
            with st.spinner("Drafting custom application..."):
                prompt = f"""
Write a tailored 200-word Statement of Purpose/Application Essay for this student applying to {target_data['Title']}.

Student Resume Context:
{resume_text}

Opportunity Details:
{target_data['Description']}
"""
                try:
                    response = generate_content_with_retry(client, prompt)
                    sop_text = response.text
                    
                    st.text_area("Generated Output", sop_text, height=220)
                    
                    # New Feature: Download Button
                    st.download_button(
                        label="📥 Download SOP as Text File",
                        data=sop_text,
                        file_name=f"{selected_opp.replace(' ', '_')}_SOP.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Failed to generate SOP: {e}")
