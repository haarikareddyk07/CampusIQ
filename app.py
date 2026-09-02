import os
import time
import re
import json
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Page setup must be called at the very top of Streamlit execution
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

# Default Dataset of Opportunities
DEFAULT_OPPORTUNITIES = [
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

# Store opportunities state in Streamlit Session State
if "opportunities" not in st.session_state:
    st.session_state["opportunities"] = DEFAULT_OPPORTUNITIES

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


def generate_content_with_retry(client, prompt, retries=3, delay=2, enable_search=False):
    """Helper function to handle API calls with exponential backoff and search grounding."""
    for attempt in range(retries):
        try:
            config = None
            if enable_search:
                config = types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
            
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config=config
            )
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            raise e


# Opportunities Section Header & Actions
st.subheader("📌 Active Opportunities")

col_fetch, col_reset = st.columns([3, 1])

with col_fetch:
    if st.button("🔍 Fetch Live Active Opportunities via AI"):
        with st.spinner("Searching the live web for currently active hackathons & internships..."):
            prompt = """
            Search for 3 real, currently active tech hackathons or student internships accepting applications.
            Output ONLY a valid JSON list of objects without markdown headers, conversational commentary, or explanation.
            
            Use EXACTLY this schema:
            [
              {
                "Title": "Opportunity Name",
                "Type": "Hackathon or Internship",
                "Deadline": "YYYY-MM-DD",
                "Domain": "Field/Skills needed",
                "Description": "Brief summary"
              }
            ]
            """
            try:
                response = generate_content_with_retry(client, prompt, enable_search=True)
                raw_text = response.text.strip()
                
                # Extract clean JSON array via Regex
                match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
                if match:
                    clean_json = match.group(0)
                    new_data = json.loads(clean_json)
                    st.session_state["opportunities"] = new_data
                    st.success("Fetched live opportunities successfully!")
                    st.rerun()
                else:
                    st.error("Could not parse JSON response from AI search. Please try clicking again.")
            except Exception as e:
                st.error(f"Failed to fetch live listings: {e}")

with col_reset:
    if st.button("🔄 Reset Default Dataset"):
        st.session_state["opportunities"] = DEFAULT_OPPORTUNITIES
        st.rerun()

# Editable DataFrame Grid
st.markdown("*Double click any cell to edit details, or click the '+' row at the bottom to manually add custom listings:*")
df = pd.DataFrame(st.session_state["opportunities"])

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key="opportunities_editor"
)

# Sync table modifications back into active dictionary
active_list = edited_df.to_dict(orient="records")

# Feature: AI Matcher & Copilot Section
st.divider()
st.subheader("⚡ AI Application Copilot")

if not active_list:
    st.warning("No opportunities available in table. Add rows above or click 'Reset Default Dataset'.")
    st.stop()

opp_titles = [op["Title"] for op in active_list if op.get("Title")]
selected_opp = st.selectbox("Select an opportunity to apply for:", opp_titles)
target_data = next((item for item in active_list if item.get("Title") == selected_opp), active_list[0])

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
Opportunity: {target_data.get('Title')} - {target_data.get('Description')} (Domains: {target_data.get('Domain')})

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
                    
                    # Extract score for visual metric display
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
Write a tailored 200-word Statement of Purpose/Application Essay for this student applying to {target_data.get('Title')}.

Student Resume Context:
{resume_text}

Opportunity Details:
{target_data.get('Description')}
"""
                try:
                    response = generate_content_with_retry(client, prompt)
                    sop_text = response.text
                    
                    st.text_area("Generated Output", sop_text, height=220)
                    
                    st.download_button(
                        label="📥 Download SOP as Text File",
                        data=sop_text,
                        file_name=f"{str(selected_opp).replace(' ', '_')}_SOP.txt",
                        mime="text/plain"
                    )
                except Exception as e:
                    st.error(f"Failed to generate SOP: {e}")
