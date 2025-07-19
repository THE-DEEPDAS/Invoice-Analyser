# Invoice Fraud Detection System

## Overview
The **Invoice Fraud Detection System** is a comprehensive tool designed to analyze invoices for potential fraud. It uses advanced techniques such as OCR (Optical Character Recognition), pattern matching, and risk scoring to identify anomalies in invoices. The system supports multiple formats, including PDFs and images, and provides detailed reports for review.

## Features
- **Invoice Analysis**: Extracts and analyzes key details such as amount, date, supplier, and GST.
- **Risk Scoring**: Assigns a risk score based on various factors like supplier validity, amount patterns, and date anomalies.
- **Trusted Supplier Tracking**: Maintains a database of trusted suppliers and their transaction history.
- **Monthly Reports**: Generates detailed monthly reports summarizing processed invoices and flagged risks.
- **Streamlit Dashboard**: Provides an interactive web interface for uploading and analyzing invoices.

## Project Structure
- `feature3_real.py`: Core logic for invoice processing, fraud detection, and risk analysis.
- `app.py`: Streamlit-based web application for user interaction and visualization.
- `fraud_detection_report.txt`: Example output report generated after processing invoices.
- `supplier_history.db`: SQLite database for storing supplier information and transaction history.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deployed-3rd-AI-feature
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required Tesseract OCR engine:
   - [Tesseract Installation Guide](https://github.com/tesseract-ocr/tesseract)

4. (Optional) Download the SpaCy language model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage
### Command-Line Interface
1. Run the main script to process invoices:
   ```bash
   python feature3_real.py
   ```
2. Follow the prompts to specify the directory containing invoices.

### Streamlit Web Application
1. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open the provided URL in your browser.
3. Upload invoices or use the sample invoice feature for analysis.

## Supported Formats
- **PDF**: Extracts text using PyPDF2.
- **Images**: Supports JPEG, PNG, and TIFF formats using Tesseract OCR.

## Reports
- **Risk Factors**: Detailed breakdown of risk scores for each invoice.
- **Monthly Summary**: Aggregated statistics for processed invoices, flagged risks, and trusted suppliers.

## Example Output
### Risk Analysis
```
File: invoice_1.pdf
Risk Score: 0.39

Risk Factors:
  - amount: 0.00
  - date: 0.80
  - supplier: 0.20
  - gst: 0.80
  - line_items: 0.50

Details:
  Amount: $71214.00
  Date: Not found
  Supplier: Tech Corp (Registered Business)
```

### Monthly Report
```
Monthly Analysis Report - 2024-12
==================================================
Total Invoices Processed: 76
Flagged for Review: 33
Total Amount Processed: ₹2,704,694.01
Trusted Suppliers: 0
```

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- **Tesseract OCR** for text extraction.
- **Streamlit** for the interactive web interface.
- **PyPDF2** for PDF text extraction.
- **SpaCy** for natural language processing.

## Contact
For questions or support, please contact the project maintainer at u23ai052@coed.svnit.ac.in
