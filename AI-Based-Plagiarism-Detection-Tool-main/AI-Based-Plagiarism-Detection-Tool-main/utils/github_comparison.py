import os
import logging
import zipfile
import tempfile
import shutil
import fnmatch
from collections import defaultdict
import re
from utils.text_comparison import get_text_similarity, find_matching_sections

# Mapping of file extensions to programming languages
EXTENSION_TO_LANGUAGE = {
    '.py': 'python',
    '.java': 'java',
    '.js': 'javascript',
    '.html': 'html',
    '.css': 'css',
    '.c': 'c',
    '.cpp': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
    '.cs': 'csharp',
    '.php': 'php',
    '.rb': 'ruby',
    '.go': 'go',
    '.ts': 'typescript',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    '.rs': 'rust',
}

# Files to ignore during comparison
IGNORE_PATTERNS = [
    '*.git*',
    '*.DS_Store',
    '*__pycache__*',
    '*.pyc',
    '*.class',
    '*.jar',
    '*.so',
    '*.dll',
    '*.exe',
    '*.o',
    '*.a',
    '*.out',
    '*.log',
    '*.sql',
    '*.db',
    '*.sqlite',
    '*.json',
    '*.csv',
    '*.xml',
    '*.yml',
    '*.yaml',
    '*.toml',
    '*.md',
    '*.txt',
    '*.pdf',
    '*.docx',
    '*.zip',
    '*.tar',
    '*.gz',
    'package-lock.json',
    'yarn.lock',
    'Pipfile.lock',
    'node_modules/*',
    'venv/*',
    'env/*',
    '.venv/*',
    '.env/*',
]

def should_ignore_file(filepath):
    """Check if a file should be ignored in the comparison"""
    for pattern in IGNORE_PATTERNS:
        if fnmatch.fnmatch(filepath, pattern):
            return True
    return False

def detect_language(filepath):
    """Detect programming language based on file extension"""
    _, ext = os.path.splitext(filepath)
    return EXTENSION_TO_LANGUAGE.get(ext.lower(), None)

def filter_files_by_language(file_list, language):
    """Filter files by programming language"""
    if language == 'auto':
        # Don't filter if auto-detect is selected
        return file_list
    
    filtered_files = []
    for filepath in file_list:
        file_language = detect_language(filepath)
        if file_language == language:
            filtered_files.append(filepath)
    
    return filtered_files if filtered_files else file_list  # Fall back to all files if none match

def extract_zip(zip_path, extract_to=None):
    """
    Extract a ZIP file to a temporary directory
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Optional directory to extract to
        
    Returns:
        str: Path to the extracted directory
    """
    try:
        if extract_to is None:
            extract_to = tempfile.mkdtemp()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        return extract_to
    except Exception as e:
        logging.error(f"Error extracting ZIP file {zip_path}: {str(e)}")
        return None

def get_source_files(repo_path, language='auto'):
    """
    Get all source code files in a repository
    
    Args:
        repo_path: Path to the repository
        language: Programming language to filter for
        
    Returns:
        list: List of file paths
    """
    source_files = []
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, repo_path)
            
            # Skip ignored files
            if should_ignore_file(rel_path):
                continue
            
            # Only include source code files
            file_language = detect_language(filepath)
            if file_language is not None:
                source_files.append(filepath)
    
    # Filter by language if specified
    if language != 'auto':
        source_files = filter_files_by_language(source_files, language)
    
    return source_files

def preprocess_code(code, language):
    """
    Preprocess code for comparison
    
    Args:
        code: Source code text
        language: Programming language
        
    Returns:
        str: Preprocessed code
    """
    # Remove comments based on language
    if language == 'python':
        # Remove Python-style comments
        code = re.sub(r'#.*', '', code)
        # Remove docstrings
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
    elif language in ['java', 'javascript', 'c', 'cpp', 'csharp']:
        # Remove C-style comments
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    
    # Remove excess whitespace
    code = re.sub(r'\s+', ' ', code).strip()
    
    return code

def compare_files(file1_path, file2_path):
    """
    Compare two source files
    
    Args:
        file1_path: Path to first file
        file2_path: Path to second file
        
    Returns:
        tuple: (similarity, matched_sections)
    """
    try:
        # Detect language
        file1_language = detect_language(file1_path)
        file2_language = detect_language(file2_path)
        
        # If languages don't match, similarity is likely low
        if file1_language != file2_language:
            return 0.1, []
        
        # Read file contents
        with open(file1_path, 'r', encoding='utf-8', errors='ignore') as f:
            code1 = f.read()
        
        with open(file2_path, 'r', encoding='utf-8', errors='ignore') as f:
            code2 = f.read()
        
        # Preprocess code
        processed_code1 = preprocess_code(code1, file1_language)
        processed_code2 = preprocess_code(code2, file2_language)
        
        # Calculate similarity
        similarity = get_text_similarity(processed_code1, processed_code2)
        
        # Find matching sections if similarity is significant
        matched_sections = []
        if similarity > 0.3:
            matched_sections = find_matching_sections(
                code1, code2, min_length=20, max_sections=5
            )
        
        return similarity, matched_sections
    except Exception as e:
        logging.error(f"Error comparing files {file1_path} and {file2_path}: {str(e)}")
        return 0.0, []

def compare_github_repos(repo1_zip, repo2_zip, language='auto'):
    """
    Compare two GitHub repositories
    
    Args:
        repo1_zip: Path to first repository ZIP
        repo2_zip: Path to second repository ZIP
        language: Programming language to filter for
        
    Returns:
        tuple: (overall_similarity, matched_sections)
    """
    temp_dirs = []
    try:
        # Extract ZIPs
        repo1_dir = extract_zip(repo1_zip)
        repo2_dir = extract_zip(repo2_zip)
        
        if not repo1_dir or not repo2_dir:
            return 0.0, []
        
        temp_dirs.extend([repo1_dir, repo2_dir])
        
        # Get source files
        repo1_files = get_source_files(repo1_dir, language)
        repo2_files = get_source_files(repo2_dir, language)
        
        if not repo1_files or not repo2_files:
            return 0.0, []
        
        # Compare all files
        file_similarities = []
        best_matches = []
        
        # Limit the number of file comparisons to avoid taking too long
        max_comparisons = 100
        comparison_count = 0
        
        for file1 in repo1_files:
            file1_rel = os.path.relpath(file1, repo1_dir)
            
            for file2 in repo2_files:
                file2_rel = os.path.relpath(file2, repo2_dir)
                
                # Skip if reached max comparisons
                comparison_count += 1
                if comparison_count > max_comparisons:
                    break
                
                # Compare the files
                similarity, sections = compare_files(file1, file2)
                
                if similarity > 0.0:
                    file_similarities.append(similarity)
                    
                    # Only keep high similarity matches
                    if similarity > 0.7:
                        best_matches.append({
                            'file1': file1_rel,
                            'file2': file2_rel,
                            'similarity': similarity,
                            'matched_sections': sections
                        })
            
            if comparison_count > max_comparisons:
                break
        
        # Calculate overall similarity
        if not file_similarities:
            return 0.0, []
        
        overall_similarity = sum(file_similarities) / len(file_similarities)
        
        # Prepare the matched sections output
        matched_sections = []
        for match in sorted(best_matches, key=lambda x: x['similarity'], reverse=True)[:5]:
            for section in match['matched_sections']:
                matched_sections.append({
                    'file1_text': f"File: {match['file1']}\n{section['file1_text']}",
                    'file2_text': f"File: {match['file2']}\n{section['file2_text']}",
                    'similarity': section['similarity']
                })
        
        return overall_similarity, matched_sections
    
    except Exception as e:
        logging.error(f"Error comparing GitHub repositories: {str(e)}")
        return 0.0, []
    
    finally:
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
