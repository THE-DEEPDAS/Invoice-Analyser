import streamlit as st
import os
import tempfile
from feature3_real import InvoiceFraudDetector
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Invoice Fraud Detection System", layout="wide")

def create_gauge_chart(score):
    """Create a gauge chart for risk score visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 33], 'color': "lightgreen"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    return fig

def main():
    st.title("Invoice Fraud Detection System")
    
    # Ensure invoices directory exists
    if not os.path.exists("invoices"):
        os.makedirs("invoices", exist_ok=True)
        
    try:
        # Initialize detector
        detector = InvoiceFraudDetector()
    except Exception as e:
        st.error(f"Error initializing the system: {str(e)}")
        st.info("Try running 'python -m spacy download en_core_web_sm' in your terminal")
        return
    
    # Enhanced upload section
    st.write("### Upload Invoices for Analysis")
    st.markdown("""
    **Supported format:** PDF files only
    - You can upload multiple invoices at once
    - Maximum file size: 200MB per file
    - Files are processed securely and not stored permanently
    """)

    # File upload with better error handling
    uploaded_files = st.file_uploader("Drop your PDF invoices here", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} files uploaded")
        
        # Create temporary directory for uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each file with error handling
            valid_files = []
            for uploaded_file in uploaded_files:
                try:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    valid_files.append(uploaded_file.name)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")

            if valid_files:
                # Show progress
                progress_text = "Analyzing invoices..."
                with st.spinner(progress_text):
                    try:
                        results = detector.analyze_invoices(temp_dir)
                        
                        # Display results section
                        st.success(f"✅ Analysis complete for {len(valid_files)} invoices")
                        st.subheader("Analysis Results")

                        for result in results:
                            with st.expander(f"Invoice: {result['filename']} - Risk Score: {result['risk_score']:.2%}"):
                                col1, col2 = st.columns([2, 1])

                                with col1:
                                    st.write("### Details")
                                    details = result['details']
                                    st.write(f"Amount: ${details['amount']:.2f}" if details['amount'] else "Amount: Not found")
                                    st.write(f"Date: {details['date']}" if details['date'] else "Date: Not found")
                                    st.write(f"Supplier: {details['supplier']}" if details['supplier'] else "Supplier: Not found")

                                    st.write("### Risk Factors")
                                    risk_df = pd.DataFrame({
                                        'Factor': result['risk_factors'].keys(),
                                        'Score': result['risk_factors'].values()
                                    })
                                    st.dataframe(risk_df)

                                with col2:
                                    st.plotly_chart(
                                        create_gauge_chart(result['risk_score']), 
                                        use_container_width=True,
                                        key=f"gauge_{result['filename']}"  # Add unique key here
                                    )

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.warning("No valid PDF files to analyze")

    else:
        st.info("👆 Upload your invoice PDFs to get started")

if __name__ == "__main__":
    main()
