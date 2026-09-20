
import streamlit as st
import requests

API_URL = "http://localhost:8000/scan"

st.set_page_config(page_title="Plagiarism Detector", layout="wide")
st.title("Explainable Semantic Plagiarism Detection")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx"])

if uploaded_file and st.button("Scan document"):
    with st.spinner("Scanning... this can take a minute (web search + SBERT scoring per passage)"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        response = requests.post(API_URL, files=files, timeout=300)

    if response.status_code != 200:
        st.error(f"Error: {response.text}")
    else:
        report = response.json()
        st.metric("Plagiarism percentage", f"{report['plagiarism_percentage']}%")
        st.write(f"{report['flagged_count']} of {report['total_passages']} passages flagged.")

        for item in report["flagged_passages"]:
            with st.expander(f"Flagged passage (fused score {item['fused_score']})"):
                st.write("**Submitted text:**", item["passage"])
                st.write("**Matched source:**", item["source_url"])
                st.write("**Source excerpt:**", item["source_excerpt"])
                col1, col2 = st.columns(2)
                col1.metric("Lexical similarity", item["lexical_score"])
                col2.metric("Semantic similarity", item["semantic_score"])
                st.info(item["explanation"])
