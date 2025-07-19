import pytesseract
from pdf2image import convert_from_path
import numpy as np
import re
from datetime import datetime, timedelta
import os
import PyPDF2  
from pathlib import Path
import mimetypes  
import json
import sqlite3
import pandas as pd
from PIL import Image
import cv2

class IndianInvoicePatterns:
    """Helper class for Indian invoice patterns"""
    def __init__(self):
        self.currency_formats = {
            'INR': [
                r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'Rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'INR\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            ],
            'USD': [
                r'\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'USD\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'US\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            ]
        }
        
        self.indian_identifiers = {
            'gst': r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}',
            'pan': r'[A-Z]{5}\d{4}[A-Z]{1}',
            'cin': r'[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}',
            'hsn': r'[0-9]{4,8}'
        }
        
        self.indian_terms = {
            'cgst': r'CGST[\s:]*(\d+(?:\.\d{2})?)',
            'sgst': r'SGST[\s:]*(\d+(?:\.\d{2})?)',
            'igst': r'IGST[\s:]*(\d+(?:\.\d{2})?)',
            'cess': r'CESS[\s:]*(\d+(?:\.\d{2})?)'
        }

class SupplierDatabase:
    def __init__(self, db_path='supplier_history.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS suppliers
                    (name TEXT PRIMARY KEY, 
                     gst_number TEXT,
                     total_transactions INTEGER,
                     average_amount REAL,
                     last_transaction_date TEXT,
                     risk_score REAL,
                     trusted BOOLEAN)''')
        conn.commit()
        conn.close()

    def update_supplier(self, supplier_info):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO suppliers VALUES 
                    (?, ?, ?, ?, ?, ?, ?)''', 
                    (supplier_info['name'],
                     supplier_info['gst_number'],
                     supplier_info['total_transactions'],
                     supplier_info['average_amount'],
                     supplier_info['last_transaction_date'],
                     supplier_info['risk_score'],
                     supplier_info['trusted']))
        conn.commit()
        conn.close()

class WatermarkDetector:
    def __init__(self):
        self.min_area = 1000
        self.max_area = 50000

    def detect_watermarks(self, image_path):
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
            
            # Find contours
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            
            # Extract text from potential watermark regions
            watermark_text = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if self.min_area < area < self.max_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = gray[y:y+h, x:x+w]
                    text = pytesseract.image_to_string(roi)
                    watermark_text.append(text.strip())
                    
            return watermark_text
        except Exception:
            return []

class IndustryAnalyzer:
    def __init__(self):
        self.industry_keywords = {
            'technology': {'software', 'hardware', 'it services', 'computing', 'technology'},
            'manufacturing': {'manufacturing', 'production', 'assembly', 'industrial', 'factory'},
            'retail': {'retail', 'store', 'shop', 'mart', 'supermarket'},
            'healthcare': {'hospital', 'medical', 'healthcare', 'pharmacy', 'clinic'},
            'construction': {'construction', 'building', 'contractor', 'infrastructure'}
        }
        
        self.industry_risk_factors = {
            'technology': 0.3,
            'manufacturing': 0.4,
            'retail': 0.35,
            'healthcare': 0.25,
            'construction': 0.45
        }

class SimpleInvoiceFraudDetector:
    def __init__(self, base_folder='invoices'):
        self.gst_pattern = r'\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}'
        self.amount_patterns = [
            r'₹\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'Total:?\s*₹?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        ]
        self.date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}'
        ]
        self.risk_weights = {
            'amount': 0.3,
            'date': 0.2,
            'supplier': 0.2,
            'gst': 0.15,
            'line_items': 0.15
        }
        self.typical_amounts = {
            'low': 5000,
            'medium': 50000,
            'high': 200000
        }
        self.known_suppliers = set() 
        self.valid_gst_prefixes = {'01', '02', '03', '04', '05', '06', '07', '08'}
        self.indian_patterns = IndianInvoicePatterns()
        self.exchange_rate = 83.0
        
        self.amount_patterns = self.indian_patterns.currency_formats
        self.supported_formats = {
            'application/pdf': self._extract_text_pdf,
            'image/jpeg': self._extract_text_image,
            'image/png': self._extract_text_image,
            'image/tiff': self._extract_text_image
        }
        mimetypes.init()
        self.supplier_db = SupplierDatabase()
        self.trusted_threshold = 0.45
        self.stats_file = 'invoice_statistics.json'
        self.load_statistics()
        self.base_folder = base_folder
        self.console_output = []  # Store outputs for terminal/web display
        self.watermark_detector = WatermarkDetector()
        self.industry_analyzer = IndustryAnalyzer()
        
        # Enhanced supplier patterns
        self.supplier_patterns = {
            'header': [
                r'From:?\s*([A-Za-z\s&.,]{3,50})',
                r'Supplier:?\s*([A-Za-z\s&.,]{3,50})',
                r'Vendor:?\s*([A-Za-z\s&.,]{3,50})',
                r'Billed By:?\s*([A-Za-z\s&.,]{3,50})',
                r'Company:?\s*([A-Za-z\s&.,]{3,50})'
            ],
            'address_block': r'([A-Za-z\s&.,]{3,50})\n.*(?:Road|Street|Avenue|Lane)',
            'letterhead': r'^([A-Za-z\s&.,]{3,50})\n'
        }

        # Add validation thresholds
        self.validation_thresholds = {
            'supplier_min_length': 3,
            'supplier_max_length': 60,  # reduced from 100
            'min_words_company': 2,
            'max_address_lines': 5
        }
        
        # Add common business words and invalid terms
        self.business_terms = {
            'prefixes': {'m/s', 'messrs', 'mr', 'mrs', 'ms', 'dr', 'company', 'co'},
            'suffixes': {'ltd', 'limited', 'llp', 'pvt', 'private', 'inc', 'corp'},
            'invalid_terms': {'please note', 'invoice', 'bill', 'tax', 'total', 'amount', 'date', 'payment', 'the trip detail page for a full tax breakdown.'}
        }

    def log_message(self, message):
        """Add message to console output and print it."""
        print(message)
        self.console_output.append(message)

    def load_statistics(self):
        try:
            with open(self.stats_file, 'r') as f:
                self.statistics = json.load(f)
        except FileNotFoundError:
            self.statistics = {
                'total_processed': 0,
                'total_flagged': 0,
                'monthly_stats': {},
                'amount_ranges': {
                    'low': 0,
                    'medium': 0,
                    'high': 0
                }
            }

    def _extract_text_pdf(self, file_path):
        """Extract text from PDF files."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return self._clean_text(text)
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {str(e)}")
            return ""

    def _preprocess_for_handwritten(self, image_path):
        """
        Apply additional image processing for handwritten or non-standard bills.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Adaptive thresholding for low contrast text
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 21, 10)
        # Morphological operations for noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img

    def _extract_text_image(self, file_path):
        """Extract text from image files using OCR with improved preprocessing."""
        try:
            # Preprocess for handwritten or non-standard format
            processed = self._preprocess_for_handwritten(file_path)
            text = pytesseract.image_to_string(processed)
            return self._clean_text(text)
        except Exception as e:
            print(f"Error extracting text from image {file_path}: {str(e)}")
            return ""

    def _clean_text(self, text):
        """Clean and normalize extracted text."""
        text = text.replace('\n\n', '\n')
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_file_type(self, file_path):
        """Detect file type using mimetypes."""
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type
            
        ext = Path(file_path).suffix.lower()
        if ext == '.pdf':
            return 'application/pdf'
        elif ext in ['.jpg', '.jpeg']:
            return 'image/jpeg'
        elif ext == '.png':
            return 'image/png'
        elif ext == '.tiff':
            return 'image/tiff'
        return None

    def process_directory(self, directory_path=None):
        """Process all supported files in directory."""
        if directory_path is None:
            directory_path = self.base_folder
            
        directory = Path(directory_path)
        results = []
        
        if not directory.exists():
            self.log_message(f"Creating directory: {directory_path}")
            directory.mkdir(parents=True, exist_ok=True)
            return results

        # Get all files in directory and subdirectories
        all_files = list(directory.rglob('*'))
        total_files = len(all_files)
        
        self.log_message(f"\nFound {total_files} files in {directory_path}")
        self.log_message("=" * 50)
        
        for index, file_path in enumerate(all_files, 1):
            if file_path.is_file():
                self.log_message(f"\nProcessing [{index}/{total_files}]: {file_path.name}")
                
                file_type = self._get_file_type(str(file_path))
                if file_type in self.supported_formats:
                    text = self.supported_formats[file_type](str(file_path))
                    if text:
                        result = self.analyze_invoice(text, str(file_path))
                        self._print_analysis(result)
                        results.append(result)
                else:
                    self.log_message(f"Skipping unsupported file type: {file_type}")

        return results

    def _print_analysis(self, result):
        """Enhanced analysis printing."""
        self.log_message("\nAnalysis Results:")
        self.log_message("-" * 30)
        self.log_message(f"File: {result['filename']}")
        self.log_message(f"Risk Score: {result['risk_score']:.2f}")
        
        self.log_message("\nRisk Factors:")
        for factor, score in result['risk_factors'].items():
            self.log_message(f"  - {factor}: {score:.2f}")
        
        self.log_message("\nDetails:")
        details = result['details']
        
        # Amount with currency
        if details.get('amount'):
            currency_symbol = '₹' if details.get('currency', 'INR') == 'INR' else '$'
            self.log_message(f"  Amount: {currency_symbol}{details['amount']:.2f}")
        else:
            self.log_message("  Amount: Not found")
            
        # Date and Supplier
        self.log_message(f"  Date: {details['date']}" if details.get('date') else "  Date: Not found")
        
        # Print supplier with validation status
        supplier = details.get('supplier')
        if supplier:
            validation_msg = ""
            if supplier in self.known_suppliers:
                validation_msg = " (Known Supplier)"
            elif any(supplier.lower().endswith(suffix) for suffix in self.business_terms['suffixes']):
                validation_msg = " (Registered Business)"
            self.log_message(f"  Supplier: {supplier}{validation_msg}")
        else:
            self.log_message("  Supplier: Not found or invalid")
        
        # Indian specific details
        if 'indian_details' in details:
            indian = details['indian_details']
            if indian.get('gst_number'):
                self.log_message(f"  GST: {indian['gst_number']}")
            
            tax_details = indian.get('tax_details', {})
            if any(tax_details.values()):
                self.log_message("\nTax Details:")
                for tax_type, value in tax_details.items():
                    if value:
                        self.log_message(f"  {tax_type.upper()}: {value}%")

    def analyze_invoice(self, text, file_path):
        """Enhanced invoice analysis with history tracking."""
        risks = {}
        
        # Extract and analyze all components
        amount, currency = self._extract_amount(text)
        indian_details = self._extract_indian_details(text)
        date = self._extract_date(text)
        supplier = self._extract_supplier(text, file_path)
        gst = self._extract_gst(text)
        items = self._extract_line_items(text)
        
        # Calculate risks
        risks['amount'] = self._check_amount_risk(amount)
        risks['date'] = self._check_date_risk(date)
        risks['supplier'] = self._check_supplier_risk(supplier)
        risks['gst'] = self._check_gst_risk(gst)
        risks['line_items'] = self._check_line_items_risk(items, amount)
        
        # Add GST-specific risk factors
        if indian_details['tax_details']['igst'] and any([
            indian_details['tax_details']['cgst'],
            indian_details['tax_details']['sgst']
        ]):
            risks['tax_consistency'] = 1.0

        final_score = sum(
            score * self.risk_weights[factor] 
            for factor, score in risks.items()
            if factor in self.risk_weights
        )

        result = {
            'filename': Path(file_path).name,
            'risk_score': final_score,
            'details': {
                'amount': amount,
                'currency': currency,
                'date': date,
                'supplier': supplier,
                'indian_details': indian_details,
                'line_items': items
            },
            'risk_factors': risks
        }
        
        # Update supplier history and statistics
        self.update_supplier_history(result)
        self.update_statistics(result)
        
        return result

    def _extract_amount(self, text):
        """Enhanced amount extraction with currency detection."""
        amounts = {}
        
        for currency, patterns in self.amount_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        amount = float(match.group(1).replace(',', ''))
                        if currency not in amounts or amount > amounts[currency]:
                            amounts[currency] = amount
                    except:
                        continue
        
        if not amounts:
            return None, None
            
        # If both currencies present, convert USD to INR
        if 'USD' in amounts and 'INR' in amounts:
            if amounts['USD'] * self.exchange_rate > amounts['INR']:
                return amounts['USD'] * self.exchange_rate, 'USD'
            return amounts['INR'], 'INR'
        
        # Return the first found amount and its currency
        currency = next(iter(amounts.keys()))
        amount = amounts[currency]
        if currency == 'USD':
            return amount * self.exchange_rate, currency
        return amount, currency

    def _check_amount_risk(self, amount):
        """Enhanced amount risk checking."""
        if not amount:
            return 0.8  # Less severe than complete failure
        
        risk = 0.0
        
        # Check for suspicious patterns
        if amount % 1000 == 0:
            risk += 0.2  # Round numbers are suspicious but not definitive
        
        # Amount range analysis
        if amount < self.typical_amounts['low']:
            risk += 0.1  # Small amounts are less risky
        elif amount > self.typical_amounts['high']:
            risk += 0.4  # Very large amounts need more scrutiny
        
        # Check for common fake amounts
        if str(amount) in {'12345.00', '98765.00', '50000.00'}:
            risk += 0.3
            
        # Digit pattern analysis
        amount_str = str(int(amount))
        if len(set(amount_str)) <= 2:  # Too many repeated digits
            risk += 0.3
        
        return min(risk, 1.0)

    def _extract_date(self, text):
        """Improved date extraction."""
        patterns = [
            r'(?:Date|Invoice Date|Bill Date)[\s:]*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'(?:Date|Invoice Date|Bill Date)[\s:]*(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b'
        ]
        
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d %B %Y', '%d %b %Y',
            '%d/%m/%y', '%d-%m-%y'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                for fmt in date_formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except:
                        continue
        return None

    def _check_date_risk(self, date):
        """Enhanced date risk checking."""
        if not date:
            return 0.8
        
        risk = 0.0
        today = datetime.now()
        
        # Future date check
        if date > today:
            days_future = (date - today).days
            if days_future > 7:  # More than a week in future
                risk += 0.6
            else:
                risk += 0.2  # Slight future dates might be okay
        
        # Past date check
        days_past = (today - date).days
        if days_past > 365:  # More than a year old
            risk += 0.7
        elif days_past > 180:  # More than 6 months old
            risk += 0.4
        elif days_past > 90:  # More than 3 months old
            risk += 0.2
            
        return min(risk, 1.0)

    def _extract_supplier(self, text, file_path=None):
        supplier = None  # ensure supplier is defined
        possible_suppliers = []
        
        # Extract using patterns
        for pattern in self.supplier_patterns['header']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                supplier = match.group(1).strip()
                if self._validate_supplier_name(supplier):
                    possible_suppliers.append((supplier, 0.9))  # High confidence for header matches

        # Extract from letterhead
        first_lines = text.split('\n')[:3]  # Check first 3 lines
        for line in first_lines:
            line = line.strip()
            if self._validate_supplier_name(line):
                possible_suppliers.append((line, 0.8))  # Good confidence for letterhead

        # Check for address block
        address_matches = re.finditer(self.supplier_patterns['address_block'], text, re.MULTILINE)
        for match in address_matches:
            supplier = match.group(1).strip()
            if self._validate_supplier_name(supplier):
                possible_suppliers.append((supplier, 0.7))  # Lower confidence for address matches

        # Check watermarks if available
        if file_path:
            try:
                watermark_text = self.watermark_detector.detect_watermarks(file_path)
                for text in watermark_text:
                    if self._validate_supplier_name(text):
                        possible_suppliers.append((text, 0.6))  # Lower confidence for watermarks
            except:
                pass

        # Select best supplier name
        if possible_suppliers:
            # Sort by confidence and length
            possible_suppliers.sort(key=lambda x: (-x[1], -len(x[0])))
            return possible_suppliers[0][0]
        
        # Fallback if standard patterns yield nothing
        if not supplier:
            # Look for a word sequence near 'name' or 'tendered by' in the text
            fallback_pattern = r'(?:name|tendered by|issuer)\s*:\s*([A-Za-z0-9\s&.,]+)'
            match = re.search(fallback_pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if self._validate_supplier_name(candidate):
                    supplier = candidate
        
        return supplier

    def _validate_supplier_name(self, name):
        """Validate supplier name."""
        if not name:
            return False
            
        name = name.strip().lower()
        
        # Basic length checks
        if len(name) < self.validation_thresholds['supplier_min_length']:
            return False
        if len(name) > self.validation_thresholds['supplier_max_length']:
            return False

        # Check for invalid terms
        for term in self.business_terms['invalid_terms']:
            if term in name.lower():
                return False

        # Must contain at least some letters
        if not any(c.isalpha() for c in name):
            return False

        # Check for business terms
        words = name.split()
        if len(words) < self.validation_thresholds['min_words_company']:
            # Single word names must have business suffix
            return any(name.endswith(suffix) for suffix in self.business_terms['suffixes'])

        # Remove common prefixes and suffixes
        for prefix in self.business_terms['prefixes']:
            if words[0].lower() == prefix:
                words = words[1:]
        for suffix in self.business_terms['suffixes']:
            if words[-1].lower() == suffix:
                words = words[:-1]

        # Check remaining words
        return len(words) > 0 and all(len(word) > 1 for word in words)

    def _check_supplier_risk(self, supplier):
        """Enhanced supplier risk checking."""
        if not supplier:
            return 0.8
        
        risk = 0.0
        
        # Validation checks
        if not self._validate_supplier_name(supplier):
            return 0.7  # High risk for invalid names
        
        # Known supplier check (reduced risk)
        if supplier in self.known_suppliers:
            risk *= 0.6
            
        # Business suffix check (reduced risk)
        if any(supplier.lower().endswith(suffix) for suffix in self.business_terms['suffixes']):
            risk *= 0.8
            
        # Industry analysis
        industry_risk = self._analyze_industry_risk(supplier)
        risk = (risk + industry_risk) / 2
        
        return min(risk, 1.0)

    def _analyze_industry_risk(self, supplier):
        """Analyze industry-specific risk."""
        supplier_lower = supplier.lower()
        
        # Check against industry keywords
        for industry, keywords in self.industry_analyzer.industry_keywords.items():
            if any(keyword in supplier_lower for keyword in keywords):
                return self.industry_analyzer.industry_risk_factors[industry]
        
        return 0.4  # Default risk for unknown industry

    def _extract_gst(self, text):
        """Extract GST number."""
        match = re.search(self.gst_pattern, text)
        return match.group() if match else None

    def _check_gst_risk(self, gst):
        """Enhanced GST number risk checking."""
        if not gst:
            return 0.8
            
        risk = 0.0
        
        # Basic format check
        if not re.match(self.gst_pattern, gst):
            return 1.0
            
        # State code check
        state_code = gst[:2]
        if state_code not in self.valid_gst_prefixes:
            risk += 0.6
            
        # Check digit patterns
        numeric_part = ''.join(c for c in gst if c.isdigit())
        if len(set(numeric_part)) <= 2:  # Too many repeated numbers
            risk += 0.4
            
        return min(risk, 1.0)

    def _extract_line_items(self, text):
        """Extract line items."""
        items = []
        lines = text.split('\n')
        pattern = r'(\d+)\s*([\w\s]+)\s*(\d+(?:\.\d{2})?)\s*(\d+(?:\.\d{2})?)'
        
        for line in lines:
            match = re.search(pattern, line)
            if match:
                qty, desc, unit_price, amount = match.groups()
                items.append({
                    'quantity': float(qty),
                    'description': desc.strip(),
                    'unit_price': float(unit_price),
                    'amount': float(amount)
                })
        return items

    def _check_line_items_risk(self, items, total_amount):
        """Enhanced line items risk checking."""
        if not items:
            return 0.8
        
        risk = 0.0
        
        # Total amount check
        items_total = sum(item['amount'] for item in items)
        if total_amount:
            difference_ratio = abs(items_total - total_amount) / total_amount
            if difference_ratio > 0.01:  # More than 1% difference
                risk += difference_ratio * 0.5  # Proportional risk
        
        # Duplicate checks
        quantities = [item['quantity'] for item in items]
        prices = [item['unit_price'] for item in items]
        
        # Check for too many identical quantities
        if len(set(quantities)) == 1 and len(quantities) > 2:
            risk += 0.3
            
        # Check for rounded prices
        if all(price % 100 == 0 for price in prices):
            risk += 0.2
            
        # Check for sequential amounts
        amounts = sorted([item['amount'] for item in items])
        if len(amounts) > 2:
            differences = [amounts[i+1] - amounts[i] for i in range(len(amounts)-1)]
            if len(set(differences)) == 1:  # All differences are the same
                risk += 0.4
                
        return min(risk, 1.0)

    def _extract_indian_details(self, text):
        """Extract India-specific invoice details."""
        details = {
            'gst_number': None,
            'pan_number': None,
            'cin_number': None,
            'hsn_codes': set(),
            'tax_details': {
                'cgst': None,
                'sgst': None,
                'igst': None,
                'cess': None
            }
        }
        
        # Extract identifiers
        for id_type, pattern in self.indian_patterns.indian_identifiers.items():
            match = re.search(pattern, text)
            if match:
                details[f'{id_type}_number'] = match.group()
        
        # Extract HSN codes
        hsn_matches = re.finditer(self.indian_patterns.indian_identifiers['hsn'], text)
        details['hsn_codes'] = {match.group() for match in hsn_matches}
        
        # Extract tax details
        for tax_type, pattern in self.indian_patterns.indian_terms.items():
            match = re.search(pattern, text)
            if match:
                details['tax_details'][tax_type] = float(match.group(1))
        
        return details

    def update_supplier_history(self, result):
        """Enhanced supplier history update."""
        if result['details']['supplier']:
            supplier = result['details']['supplier']
            gst = result['details'].get('indian_details', {}).get('gst_number')
            
            # Get existing supplier data or create new
            conn = sqlite3.connect(self.supplier_db.db_path)
            c = conn.cursor()
            c.execute('SELECT * FROM suppliers WHERE name = ?', (supplier,))
            existing = c.fetchone()
            
            if existing:
                total_trans = existing[2] + 1
                avg_amount = (existing[3] * existing[2] + result['details']['amount']) / total_trans
                risk_score = (existing[5] * existing[2] + result['risk_score']) / total_trans
                trusted = risk_score < self.trusted_threshold
            else:
                total_trans = 1
                avg_amount = result['details']['amount']
                risk_score = result['risk_score']
                trusted = risk_score < self.trusted_threshold

            supplier_info = {
                'name': supplier,
                'gst_number': gst,
                'total_transactions': total_trans,
                'average_amount': avg_amount,
                'last_transaction_date': datetime.now().isoformat(),
                'risk_score': risk_score,
                'trusted': trusted
            }
            
            # Additional industry analysis
            supplier_lower = supplier.lower()
            detected_industry = None
            for industry, keywords in self.industry_analyzer.industry_keywords.items():
                if any(keyword in supplier_lower for keyword in keywords):
                    detected_industry = industry
                    break
            
            supplier_info.update({
                'industry': detected_industry,
                'last_transaction_amount': result['details']['amount'],
                'average_risk_score': risk_score,
                'transaction_frequency': total_trans / 30  # transactions per month
            })
            
            # Update trusted status based on comprehensive analysis
            is_trusted = (
                risk_score < self.trusted_threshold and
                total_trans >= 3 and  # minimum transaction history
                supplier_info['transaction_frequency'] > 0.5  # regular transactions
            )
            
            supplier_info['trusted'] = is_trusted
            self.supplier_db.update_supplier(supplier_info)

    def update_statistics(self, result):
        """Update processing statistics."""
        self.statistics['total_processed'] += 1
        if result['risk_score'] > self.trusted_threshold:
            self.statistics['total_flagged'] += 1

        # Update monthly stats
        month_key = datetime.now().strftime('%Y-%m')
        if month_key not in self.statistics['monthly_stats']:
            self.statistics['monthly_stats'][month_key] = {
                'processed': 0,
                'flagged': 0,
                'total_amount': 0
            }
        
        monthly = self.statistics['monthly_stats'][month_key]
        monthly['processed'] += 1
        monthly['total_amount'] += result['details'].get('amount', 0)
        if result['risk_score'] > self.trusted_threshold:
            monthly['flagged'] += 1

        # Save statistics
        with open(self.stats_file, 'w') as f:
            json.dump(self.statistics, f, indent=4)

    def generate_report(self, result):
        """Fixed report generation."""
        report = []
        report.append(f"Risk Score: {result['risk_score']:.2f}")
        report.append("\nDetails:")
        details = result['details']
        
        # Amount with currency
        if details['amount']:
            currency_symbol = '₹' if details.get('currency', 'INR') == 'INR' else '$'
            report.append(f"  Amount: {currency_symbol}{details['amount']:.2f}")
        
        # Add other basic details
        if details.get('date'):
            report.append(f"  Date: {details['date']}")
        if details.get('supplier'):
            report.append(f"  Supplier: {details['supplier']}")
        
        # Indian specific details - with safe access
        if 'indian_details' in details:
            indian = details['indian_details']
            if indian.get('gst_number'):
                report.append(f"  GST Number: {indian['gst_number']}")
            if indian.get('pan_number'):
                report.append(f"  PAN Number: {indian['pan_number']}")
            
            # Tax details - with safe access
            tax_details = indian.get('tax_details', {})
            if tax_details.get('cgst'):
                report.append(f"  CGST: {tax_details['cgst']}%")
            if tax_details.get('sgst'):
                report.append(f"  SGST: {tax_details['sgst']}%")
            if tax_details.get('igst'):
                report.append(f"  IGST: {tax_details['igst']}%")
        
        return "\n".join(report)

    def generate_monthly_report(self):
        """Generate monthly analysis report."""
        month_key = datetime.now().strftime('%Y-%m')
        monthly_stats = self.statistics['monthly_stats'].get(month_key, {})
        
        report = f"\nMonthly Analysis Report - {month_key}\n"
        report += "=" * 50 + "\n"
        report += f"Total Invoices Processed: {monthly_stats.get('processed', 0)}\n"
        report += f"Flagged for Review: {monthly_stats.get('flagged', 0)}\n"
        report += f"Total Amount Processed: ₹{monthly_stats.get('total_amount', 0):,.2f}\n"
        
        # Get trusted suppliers
        conn = sqlite3.connect(self.supplier_db.db_path)
        df = pd.read_sql_query('SELECT * FROM suppliers WHERE trusted = 1', conn)
        conn.close()
        
        report += f"\nTrusted Suppliers: {len(df)}\n"
        if not df.empty:
            report += "\nTop 5 Trusted Suppliers by Transaction Volume:\n"
            top_suppliers = df.nlargest(5, 'total_transactions')
            for _, row in top_suppliers.iterrows():
                report += f"  - {row['name']}: {row['total_transactions']} transactions\n"
        
        return report

def main(test_mode=False):
    try:
        # Initialize detector
        detector = SimpleInvoiceFraudDetector()
        
        # Get invoice directory from user or use default
        if not test_mode:
            invoice_dir = input("Enter invoice directory path (or press Enter for 'invoices'): ").strip()
            if not invoice_dir:
                invoice_dir = detector.base_folder
        else:
            invoice_dir = detector.base_folder
        
        detector.log_message(f"\nStarting analysis of invoices in: {invoice_dir}")
        results = detector.process_directory(invoice_dir)
        
        # Generate and save report
        if results:
            detector.log_message(f"\nProcessed {len(results)} files successfully")
            
            # Generate monthly report
            monthly_report = detector.generate_monthly_report()
            detector.log_message("\nMonthly Summary:")
            detector.log_message(monthly_report)
            
            # Show trusted suppliers
            conn = sqlite3.connect(detector.supplier_db.db_path)
            df = pd.read_sql_query('SELECT name, total_transactions FROM suppliers WHERE trusted = 1', conn)
            conn.close()
            
            if not df.empty:
                detector.log_message("\nTrusted Suppliers:")
                for _, row in df.iterrows():
                    detector.log_message(f"  - {row['name']} ({row['total_transactions']} transactions)")
            
            # Save reports to files
            report_path = Path(invoice_dir) / "fraud_detection_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(detector.console_output))
            
            detector.log_message(f"\nDetailed report saved to: {report_path}")
        else:
            detector.log_message("\nNo valid invoices found for analysis")
            
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        if test_mode:
            raise

if __name__ == "__main__":
    main(test_mode=False)  # Set to True for testing