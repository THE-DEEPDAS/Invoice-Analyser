import streamlit as st
# Set page config first, before any other Streamlit commands
st.set_page_config(page_title="Invoice Analyser", page_icon="📊", layout="wide")

import os
from pathlib import Path
import subprocess
import sys

# Handle imports with error messages
try:
    import cv2
except ImportError:
    st.error("Error loading OpenCV. Please check system dependencies.")
    
try:
    from model import SimpleInvoiceFraudDetector
except ImportError as e:
    st.error(f"Error loading model: {str(e)}")

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import PyPDF2
from datetime import datetime

def install_tesseract():
    try:
        if sys.platform.startswith('linux'):
            subprocess.run(['apt-get', 'update'])
            subprocess.run(['apt-get', 'install', '-y', 'tesseract-ocr'])
    except Exception as e:
        st.error(f"Error installing Tesseract: {e}")

def install_system_dependencies():
    """Install required system packages"""
    try:
        if sys.platform.startswith('linux'):
            subprocess.run(['apt-get', 'update'], check=True)
            subprocess.run(['apt-get', 'install', '-y', 
                          'python3-opencv', 
                          'tesseract-ocr',
                          'libgl1',
                          'poppler-utils'], check=True)
    except Exception as e:
        st.error(f"Error installing system dependencies: {e}")

def main():
    install_system_dependencies()
    install_tesseract()
    
    # Sidebar with disclaimer
    with st.sidebar:
        st.title("ℹ️ About")
        st.info(
            "This application is powered by AI and machine learning models. "
            "The analysis provided should be used as a supplementary tool and not as "
            "the sole basis for decision-making. Results may not be 100% accurate."
        )
        st.warning(
            "**Disclaimer**: This is an AI-powered tool. Always verify results "
            "manually for critical business decisions. The creators are not "
            "liable for any decisions made based on this analysis."
        )
        
    # Main content
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.title("📊 Invoice Analyser")
        st.markdown("---")
        
    # File upload section with better styling
    st.markdown("""
        <style>
        .upload-section {
            border: 2px dashed #4e8df5;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "📎 Drop your invoice files here (PDF, JPG, PNG, TIFF)",
            type=["pdf","jpg","jpeg","png","tiff"],
            accept_multiple_files=True,
            help="You can select multiple files at once"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_files:
        detector = SimpleInvoiceFraudDetector()
        results = []
        temp_dir = Path("temp_upload")
        temp_dir.mkdir(exist_ok=True)
        
        # Progress tracking
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"⏳ Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                temp_path = temp_dir / uploaded_file.name
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                try:
                    result = detector.analyze_invoice(
                        detector.supported_formats[detector._get_file_type(str(temp_path))](str(temp_path)), 
                        str(temp_path)
                    )
                    results.append(result)
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ Analysis Complete!")
        st.markdown("---")
        
        # Results display
        st.subheader("📑 Analysis Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Files Processed", len(results))
        with col2:
            avg_risk = sum(r['risk_score'] for r in results) / len(results)
            st.metric("Average Risk Score", f"{avg_risk:.2f}")
        with col3:
            high_risk = sum(1 for r in results if r['risk_score'] > 0.7)
            st.metric("High Risk Files", high_risk)
        
        # Detailed results
        for result in results:
            with st.expander(f"📄 {Path(result['filename']).name}"):
                risk_score = result['risk_score']
                
                # Risk score with color coding
                risk_color = (
                    "🟢 Low" if risk_score < 0.4 
                    else "🟡 Medium" if risk_score < 0.7 
                    else "🔴 High"
                )
                
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"### Risk Level: {risk_color}")
                    st.markdown(f"### Score: {risk_score:.2f}")
                
                with col2:
                    st.json(result['details'])
                
                if risk_score > 0.7:
                    st.warning("⚠️ This invoice requires immediate attention!")

if __name__ == "__main__":
    main()