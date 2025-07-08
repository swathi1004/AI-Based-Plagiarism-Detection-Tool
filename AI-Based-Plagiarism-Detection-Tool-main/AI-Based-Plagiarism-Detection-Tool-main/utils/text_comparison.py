import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import re

def preprocess_text(text):
    """
    Preprocess text by converting to lowercase, removing special characters, and extra whitespace
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_file_content(file_path):
    """
    Read and return the content of a text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {str(e)}")
        return ""

def get_text_similarity(text1, text2):
    """
    Calculate cosine similarity between two text documents using TF-IDF
    """
    try:
        # Preprocess the texts
        preprocessed_text1 = preprocess_text(text1)
        preprocessed_text2 = preprocess_text(text2)
        
        # Create TF-IDF vectorizer and transform documents
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text1, preprocessed_text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        logging.error(f"Error calculating text similarity: {str(e)}")
        return 0.0

def find_matching_sections(text1, text2, min_length=40, max_sections=10):
    """
    Find and return the matching sections between two texts
    """
    try:
        # Split texts into sentences
        sentences1 = re.split(r'(?<=[.!?])\s+', text1)
        sentences2 = re.split(r'(?<=[.!?])\s+', text2)
        
        # Find matching sections using difflib
        matcher = difflib.SequenceMatcher(None, sentences1, sentences2)
        matching_blocks = matcher.get_matching_blocks()
        
        matched_sections = []
        for match in matching_blocks:
            i, j, size = match
            if size > 0:
                # Extract matching text sections
                matched_text1 = ' '.join(sentences1[i:i+size])
                matched_text2 = ' '.join(sentences2[j:j+size])
                
                # Only include if they're substantial enough
                if len(matched_text1) >= min_length:
                    # Calculate similarity for this section
                    section_similarity = get_text_similarity(matched_text1, matched_text2)
                    
                    matched_sections.append({
                        'file1_text': matched_text1,
                        'file2_text': matched_text2,
                        'similarity': section_similarity
                    })
        
        # Sort by similarity (highest first) and limit the number of sections
        matched_sections.sort(key=lambda x: x['similarity'], reverse=True)
        return matched_sections[:max_sections]
    except Exception as e:
        logging.error(f"Error finding matching sections: {str(e)}")
        return []

def compare_text_files(file1_path, file2_path, text1=None, text2=None):
    """
    Compare two text files and return the similarity score and matching sections
    
    Args:
        file1_path: Path to the first file
        file2_path: Path to the second file
        text1: Optional pre-loaded text content of file1
        text2: Optional pre-loaded text content of file2
        
    Returns:
        tuple: (similarity_score, matched_sections)
    """
    try:
        # Get file contents if not provided
        if text1 is None:
            text1 = get_file_content(file1_path)
        if text2 is None:
            text2 = get_file_content(file2_path)
        
        # Calculate overall similarity
        similarity = get_text_similarity(text1, text2)
        
        # Find matching sections
        matched_sections = find_matching_sections(text1, text2)
        
        return similarity, matched_sections
    except Exception as e:
        logging.error(f"Error comparing text files: {str(e)}")
        return 0.0, []
