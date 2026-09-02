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

# Initialize default opportunity list in Session State
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


def generate_content_with_retry(client, prompt, retries=3, delay=2, tools=None):
    """Helper function with retry logic and tool support."""
    for attempt in range(retries):
        try:
            return client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
                config={"tools": tools} if tools else None
            )
        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            raise e


# Display and Manage Opportunities Section
st.subheader("📌 Active Opportunities")

col_fetch, col_reset = st.columns([3, 1])

with col_fetch:
    if st.button("🔍 Fetch Live Active Opportunities via AI"):
        with st.spinner("Searching the live web for currently active hackathons & internships..."):
            prompt = """
            Search the web for current active tech hackathons, open source programs, or student internships accepting applications in the next 30-60 days.
            
            Return a JSON array of 3 top opportunities using EXACTLY this schema:
            [
              {
                "Title": "Name",
                "Type": "Hackathon or Internship",
                "Deadline": "YYYY-MM-DD",
                "Domain": "Relevant skills",
                "Description": "Brief overview"
              }
            ]
            Provide ONLY raw valid JSON text without markdown codeblocks or extra prose.
            """
            try:
                # Grounded with Google Search enabled
                response = generate_content_with_retry(client, prompt, tools=[{"type": "google_search"}])
                raw_text = response.text.replace("```json", "").replace("```", "").strip()
                import json
                new_data = json.loads(raw_text)
                st.session_state["opportunities"] = new_data
                st.success("Successfully fetched live opportunities!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to fetch live listings: {e}")

with col_reset:
    if st.button("🔄 Reset Default Dataset"):
        st.session_state["opportunities"] = DEFAULT_OPPORTUNITIES
        st.rerun()

# Editable DataFrame Table (Option 1 + Option 2 Combined)
st.markdown("*Double click any cell to edit details, or click the '+' row at the bottom to manually add custom listings:*")
df = pd.DataFrame(st.session_state["opportunities"])

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    use_container_width=True,
    key="opportunities_editor"
)

# Convert edited data back into a list of dicts for model evaluation
active_list = edited_df.to_dict(orient="records")

# Feature: AI Matcher & Copilot
st.divider()
st.subheader("⚡ AI Application Copilot")

if not active_list:
    st.warning("No opportunities available. Add rows above or click 'Reset Default Dataset'.")
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
