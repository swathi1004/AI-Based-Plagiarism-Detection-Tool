import os
import logging
import PyPDF2
from io import StringIO
import pdfplumber
from utils.enhanced_text_comparison import (
    get_text_similarity_tfidf, 
    get_text_similarity_bert, 
    find_matching_sections,
    BERT_AVAILABLE
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Try to import OCR functionality
try:
    import easyocr
    reader = easyocr.Reader(['en'])  # Initialize once (slow)
    OCR_AVAILABLE = True
    logger.info("EasyOCR initialized successfully")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("easyocr not available, OCR functionality disabled")

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
                page_text = pdf_reader.pages[page_num].extract_text() or ""
                text += page_text + "\n"
        
        # If PyPDF2 fails to extract meaningful text, try pdfplumber
        if text.strip() == "":
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
        
        # If both text extraction methods fail, it might be a scanned PDF, try OCR
        if text.strip() == "" and OCR_AVAILABLE:
            try:
                text = extract_text_with_easyocr(pdf_path)
                logger.info(f"Used OCR to extract text from {pdf_path}")
            except Exception as e:
                logger.error(f"OCR extraction failed: {str(e)}")
                
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return ""

def extract_text_with_easyocr(file_path):
    """
    Extract text from scanned PDFs/images using EasyOCR.
    
    Args:
        file_path: Path to the PDF or image file
        
    Returns:
        str: Extracted text using OCR
    """
    if not OCR_AVAILABLE:
        logger.error("EasyOCR not available")
        return ""
        
    try:
        results = reader.readtext(file_path, paragraph=True)
        return "\n".join([res[1] for res in results])
    except Exception as e:
        logger.error(f"EasyOCR failed: {str(e)}")
        return ""

def compare_pdf_files(pdf1_path, pdf2_path, use_bert=True):
    """
    Compare two PDF files by extracting their text content and using text comparison methods
    
    Args:
        pdf1_path: Path to the first PDF file
        pdf2_path: Path to the second PDF file
        use_bert: Whether to use BERT for similarity calculation
        
    Returns:
        tuple: (similarity_score, matched_sections)
    """
    try:
        # Extract text from both PDFs
        text1 = extract_text_from_pdf(pdf1_path)
        text2 = extract_text_from_pdf(pdf2_path)
        
        # If either extraction failed, return zero similarity
        if not text1 or not text2:
            logger.warning("Text extraction failed for one or both PDFs")
            return 0.0, []
        
        # Use text comparison methods on the extracted text
        if use_bert and BERT_AVAILABLE:
            try:
                similarity = get_text_similarity_bert(text1, text2)
            except Exception as e:
                logger.error(f"BERT similarity failed for PDFs, falling back to TF-IDF: {e}")
                similarity = get_text_similarity_tfidf(text1, text2)
        else:
            similarity = get_text_similarity_tfidf(text1, text2)
            
        matched_sections = find_matching_sections(text1, text2)
        
        return similarity, matched_sections
    except Exception as e:
        logger.error(f"Error comparing PDF files: {str(e)}")
        return 0.0, []

def compare_text_to_pdf(text_path, pdf_path, use_bert=True):
    """
    Compare a text file to a PDF file
    
    Args:
        text_path: Path to the text file
        pdf_path: Path to the PDF file
        use_bert: Whether to use BERT for similarity calculation
        
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
            logger.warning(f"Text extraction failed for PDF: {pdf_path}")
            return 0.0, []
        
        # Use text comparison methods
        if use_bert and BERT_AVAILABLE:
            try:
                similarity = get_text_similarity_bert(text_content, pdf_text)
            except Exception as e:
                logger.error(f"BERT similarity failed for text-PDF, falling back to TF-IDF: {e}")
                similarity = get_text_similarity_tfidf(text_content, pdf_text)
        else:
            similarity = get_text_similarity_tfidf(text_content, pdf_text)
        
        matched_sections = find_matching_sections(text_content, pdf_text)
        
        return similarity, matched_sections
    except Exception as e:
        logger.error(f"Error comparing text to PDF: {str(e)}")
        return 0.0, []

# For direct testing
if __name__ == "__main__":
    # Test with sample PDFs if available
    pdf_test_folder = "test_pdfs"
    
    if os.path.exists(pdf_test_folder):
        pdf_files = [f for f in os.listdir(pdf_test_folder) if f.endswith('.pdf')]
        
        if len(pdf_files) >= 2:
            pdf1 = os.path.join(pdf_test_folder, pdf_files[0])
            pdf2 = os.path.join(pdf_test_folder, pdf_files[1])
            
            print(f"Testing extraction from {pdf1}")
            text1 = extract_text_from_pdf(pdf1)
            print(f"Extracted {len(text1)} characters")
            print(text1[:200] + "..." if len(text1) > 200 else text1)
            
            print(f"\nTesting extraction from {pdf2}")
            text2 = extract_text_from_pdf(pdf2)
            print(f"Extracted {len(text2)} characters")
            print(text2[:200] + "..." if len(text2) > 200 else text2)
            
            print("\nComparing PDFs:")
            similarity, sections = compare_pdf_files(pdf1, pdf2)
            print(f"Similarity: {similarity:.4f}")
            print(f"Found {len(sections)} matching sections")
        else:
            print(f"Not enough PDF files found in {pdf_test_folder}")
    else:
        print(f"{pdf_test_folder} directory not found")