import os
import logging
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
import difflib

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
    
    # Initialize model (this will be skipped if import fails)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    BERT_AVAILABLE = True
    logger.info("Sentence-BERT model loaded successfully")
except ImportError:
    BERT_AVAILABLE = False
    logger.warning("sentence-transformers not available, falling back to TF-IDF")

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

def get_text_similarity_tfidf(text1, text2):
    """
    Traditional method: Calculate cosine similarity between two text documents using TF-IDF
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
        logging.error(f"Error calculating text similarity with TF-IDF: {str(e)}")
        return 0.0

def get_text_similarity_bert(text1, text2):
    """
    Enhanced method: Calculate semantic similarity using Sentence-BERT
    """
    if not BERT_AVAILABLE:
        logging.warning("BERT not available, using TF-IDF instead")
        return get_text_similarity_tfidf(text1, text2)
    
    try:
        # Truncate very long texts to avoid memory issues
        if len(text1) > 10000:
            text1 = text1[:10000]
        if len(text2) > 10000:
            text2 = text2[:10000]
            
        # Generate embeddings
        emb1 = model.encode(text1, convert_to_tensor=True)
        emb2 = model.encode(text2, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(emb1, emb2).item()
        return similarity
    except Exception as e:
        logging.error(f"Error calculating text similarity with BERT: {str(e)}")
        # Fallback to traditional method
        return get_text_similarity_tfidf(text1, text2)

def extract_stylometric_features(text):
    """
    Extract writing style features from text
    
    Args:
        text: The input text
        
    Returns:
        dict: Dictionary of stylometric features
    """
    # Handle empty text
    if not text or len(text.strip()) == 0:
        return {
            'avg_word_length': 0,
            'punctuation_density': 0,
            'avg_sentence_length': 0,
            'function_word_ratio': 0,
            'uppercase_ratio': 0
        }
    
    # Tokenize
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    
    # Count various elements
    total_chars = sum(len(w) for w in words) if words else 0
    total_punctuation = sum(1 for c in text if c in ',.;:!?"()')
    
    # Function words (common words that don't carry strong meaning)
    function_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
        'down', 'in', 'out', 'on', 'off', 'over', 'under', 'he', 'she', 'it',
        'they', 'we', 'who', 'what', 'where', 'when', 'why', 'how', 'all',
        'any', 'both', 'each', 'few', 'more', 'most', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'as'
    }
    
    function_word_count = sum(1 for w in words if w in function_words)
    uppercase_count = sum(1 for c in text if c.isupper())
    
    # Calculate metrics
    avg_word_length = total_chars / len(words) if words else 0
    punctuation_density = total_punctuation / len(text) if text else 0
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    function_word_ratio = function_word_count / len(words) if words else 0
    uppercase_ratio = uppercase_count / len(text) if text else 0
    
    return {
        'avg_word_length': avg_word_length,
        'punctuation_density': punctuation_density,
        'avg_sentence_length': avg_sentence_length,
        'function_word_ratio': function_word_ratio,
        'uppercase_ratio': uppercase_ratio
    }

def get_stylometric_similarity(text1, text2):
    """
    Calculate stylometric similarity between two texts
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    features1 = extract_stylometric_features(text1)
    features2 = extract_stylometric_features(text2)
    
    # Calculate normalized Euclidean distance between feature vectors
    squared_diff_sum = 0
    for key in features1:
        # Skip if feature is 0 in both texts
        if features1[key] == 0 and features2[key] == 0:
            continue
            
        max_val = max(abs(features1[key]), abs(features2[key]))
        if max_val > 0:  # Avoid division by zero
            norm1 = features1[key] / max_val
            norm2 = features2[key] / max_val
            squared_diff_sum += (norm1 - norm2) ** 2
    
    # Convert distance to similarity (1 = identical, 0 = completely different)
    distance = np.sqrt(squared_diff_sum)
    similarity = 1 / (1 + distance)  # Transform to 0-1 range using sigmoid-like function
    
    return similarity

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
                    try:
                        if BERT_AVAILABLE:
                            section_similarity = get_text_similarity_bert(matched_text1, matched_text2)
                        else:
                            section_similarity = get_text_similarity_tfidf(matched_text1, matched_text2)
                    except:
                        section_similarity = get_text_similarity_tfidf(matched_text1, matched_text2)
                    
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

def compare_text_files(file1_path, file2_path, text1=None, text2=None, use_bert=True):
    """
    Compare two text files and return the similarity score and matching sections
    
    Args:
        file1_path: Path to the first file
        file2_path: Path to the second file
        text1: Optional pre-loaded text content of file1
        text2: Optional pre-loaded text content of file2
        use_bert: Whether to use BERT for similarity or fallback to TF-IDF
        
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
        if use_bert and BERT_AVAILABLE:
            try:
                semantic_similarity = get_text_similarity_bert(text1, text2)
            except Exception as e:
                logging.error(f"BERT similarity failed, falling back to TF-IDF: {e}")
                semantic_similarity = get_text_similarity_tfidf(text1, text2)
        else:
            semantic_similarity = get_text_similarity_tfidf(text1, text2)
        
        # Get stylometric similarity as an additional signal
        style_similarity = get_stylometric_similarity(text1, text2)
        
        # Combine semantic and stylometric similarity with more weight on semantic
        similarity = 0.8 * semantic_similarity + 0.2 * style_similarity
        
        # Find matching sections
        matched_sections = find_matching_sections(text1, text2)
        
        return similarity, matched_sections
    except Exception as e:
        logging.error(f"Error comparing text files: {str(e)}")
        return 0.0, []

# For direct testing
if __name__ == "__main__":
    # Test the module
    sample_text1 = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence 
    displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, 
    which refers to any system that perceives its environment and takes actions that maximize its chance of achieving 
    its goals.
    """
    
    sample_text2 = """
    Artificial intelligence or AI refers to intelligence shown by machines, as opposed to natural intelligence 
    exhibited by animals and humans. Research in AI has been defined as studying intelligent agents, 
    which are systems that observe their environment and take actions to maximize their chances of achieving 
    their goals.
    """
    
    sample_text3 = """
    Climate change is the long-term alteration of temperature and typical weather patterns in a place. 
    Climate change could refer to a particular location or the planet as a whole. Climate change may cause 
    weather patterns to be less predictable.
    """
    
    print("Testing TF-IDF similarity:")
    sim_tfidf = get_text_similarity_tfidf(sample_text1, sample_text2)
    print(f"Similar texts: {sim_tfidf:.4f}")
    
    sim_diff_tfidf = get_text_similarity_tfidf(sample_text1, sample_text3)
    print(f"Different texts: {sim_diff_tfidf:.4f}")
    
    if BERT_AVAILABLE:
        print("\nTesting BERT similarity:")
        sim_bert = get_text_similarity_bert(sample_text1, sample_text2)
        print(f"Similar texts: {sim_bert:.4f}")
        
        sim_diff_bert = get_text_similarity_bert(sample_text1, sample_text3)
        print(f"Different texts: {sim_diff_bert:.4f}")
    
    print("\nTesting stylometric similarity:")
    sim_style = get_stylometric_similarity(sample_text1, sample_text2)
    print(f"Similar texts: {sim_style:.4f}")
    
    sim_diff_style = get_stylometric_similarity(sample_text1, sample_text3)
    print(f"Different texts: {sim_diff_style:.4f}")