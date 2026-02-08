import streamlit as st
import pandas as pd
import sys
import os

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analyzer.parser import analyze_file

st.set_page_config(page_title="ROCm Bridge", layout="wide")

st.title("🌉 The ROCm Bridge")
st.markdown("### Automated CUDA-to-HIP Optimization Engine")

uploaded_file = st.file_uploader("Upload CUDA File (.cu)", type=["cu"])

if uploaded_file is not None:
    # Save temp file
    with open("temp_upload.cu", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success("File Uploaded!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Code")
        st.code(uploaded_file.getvalue().decode("utf-8"), language="cpp")

    # Run Analysis
    issues = analyze_file("temp_upload.cu")
    
    with col2:
        st.subheader("Analysis Report")
        if issues:
            for i in issues:
                st.error(f"Line {i['line']}: {i['rule']['message']}")
                if i['rule']['severity'] == 'HIGH':
                    st.warning("💡 Recommendation: Upgrade to Wave64 (AMD Native)")
        else:
            st.success("No Static Issues Found!")

