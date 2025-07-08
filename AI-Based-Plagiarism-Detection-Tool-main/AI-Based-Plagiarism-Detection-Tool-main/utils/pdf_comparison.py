import os
import logging
import PyPDF2
from io import StringIO
import pdfplumber
from utils.text_comparison import compare_text_files, get_text_similarity, find_matching_sections

def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file using PyPDF2 and pdfplumber as fallback
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # First try with PyPDF2
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() + "\n"
        
        # If PyPDF2 fails to extract meaningful text, try pdfplumber
        if text.strip() == "":
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def compare_pdf_files(pdf1_path, pdf2_path):
    """
    Compare two PDF files by extracting their text content and using text comparison methods
    
    Args:
        pdf1_path: Path to the first PDF file
        pdf2_path: Path to the second PDF file
        
    Returns:
        tuple: (similarity_score, matched_sections)
    """
    try:
        # Extract text from both PDFs
        text1 = extract_text_from_pdf(pdf1_path)
        text2 = extract_text_from_pdf(pdf2_path)
        
        # If either extraction failed, return zero similarity
        if not text1 or not text2:
            return 0.0, []
        
        # Use text comparison methods on the extracted text
        similarity = get_text_similarity(text1, text2)
        matched_sections = find_matching_sections(text1, text2)
        
        return similarity, matched_sections
    except Exception as e:
        logging.error(f"Error comparing PDF files: {str(e)}")
        return 0.0, []

def compare_text_to_pdf(text_path, pdf_path):
    """
    Compare a text file to a PDF file
    
    Args:
        text_path: Path to the text file
        pdf_path: Path to the PDF file
        
    Returns:
        tuple: (similarity_score, matched_sections)
    """
    try:
        # Extract text from the PDF
        pdf_text = extract_text_from_pdf(pdf_path)
        
        # Read the text file
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as file:
            text_content = file.read()
        
        # If PDF extraction failed, return zero similarity
        if not pdf_text:
            return 0.0, []
        
        # Use text comparison methods
        similarity = get_text_similarity(text_content, pdf_text)
        matched_sections = find_matching_sections(text_content, pdf_text)
        
        return similarity, matched_sections
    except Exception as e:
        logging.error(f"Error comparing text to PDF: {str(e)}")
        return 0.0, []
