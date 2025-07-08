import os
import logging
import tempfile
import uuid
import json
import re
from difflib import SequenceMatcher
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename
# Try to import enhanced modules first, fallback to basic modules if not available
try:
    from utils.enhanced_text_comparison import compare_text_files
    from utils.enhanced_pdf_comparison import compare_pdf_files, extract_text_from_pdf
    from utils.enhanced_github_comparison import compare_github_repos
    USING_ENHANCED_MODELS = True
    logging.info("Using enhanced ML-based comparison modules")
except ImportError:
    from utils.text_comparison import compare_text_files
    from utils.pdf_comparison import compare_pdf_files, extract_text_from_pdf
    from utils.github_comparison import compare_github_repos
    USING_ENHANCED_MODELS = False
    logging.info("Using basic comparison modules (ML features not available)")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "universal_plagiarism_checker_secret")

# Configure upload settings
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'zip', 'py', 'java', 'cpp', 'c', 'js', 'html', 'css'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect-plagiarism', methods=['POST'])
def detect_plagiarism():
    comparison_type = request.form.get('comparison_type', 'text_text')
    programming_language = request.form.get('programming_language', 'auto')
    
    # Check if files were uploaded
    if 'file1' not in request.files or 'file2' not in request.files:
        flash('Two files are required for comparison', 'error')
        return redirect(url_for('index'))
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    # Check if file selection was empty
    if file1.filename == '' or file2.filename == '':
        flash('Please select two files for comparison', 'error')
        return redirect(url_for('index'))
    
    # Check if files are allowed
    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        flash('Invalid file type. Please upload supported file types.', 'error')
        return redirect(url_for('index'))
    
    try:
        # Create unique filenames to avoid collisions
        filename1 = secure_filename(str(uuid.uuid4()) + '_' + file1.filename)
        filename2 = secure_filename(str(uuid.uuid4()) + '_' + file2.filename)
        
        filepath1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        filepath2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        # Save files
        file1.save(filepath1)
        file2.save(filepath2)
        
        # Process files based on comparison type
        result = {}
        matched_sections = []
        
        if comparison_type == 'text_text':
            similarity, matched_sections = compare_text_files(filepath1, filepath2)
            result = {
                'similarity': similarity,
                'matched_sections': matched_sections
            }
        elif comparison_type == 'pdf_pdf':
            similarity, matched_sections = compare_pdf_files(filepath1, filepath2)
            result = {
                'similarity': similarity,
                'matched_sections': matched_sections
            }
        elif comparison_type == 'text_pdf':
            text_content = ""
            with open(filepath1, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            pdf_text = extract_text_from_pdf(filepath2)
            similarity, matched_sections = compare_text_files(filepath1, filepath2, text1=text_content, text2=pdf_text)
            result = {
                'similarity': similarity,
                'matched_sections': matched_sections
            }
        elif comparison_type == 'github_github':
            similarity, matched_sections = compare_github_repos(filepath1, filepath2, programming_language)
            result = {
                'similarity': similarity,
                'matched_sections': matched_sections
            }
        else:
            flash('Invalid comparison type selected', 'error')
            return redirect(url_for('index'))
        
        # Clean up temporary files
        os.remove(filepath1)
        os.remove(filepath2)
        
        # Store results in session for the results page
        result_data = {
            'similarity': float(result['similarity']),
            'matched_sections': result['matched_sections'],
            'file1_name': file1.filename,
            'file2_name': file2.filename,
            'comparison_type': comparison_type,
            'using_enhanced_models': USING_ENHANCED_MODELS
        }
        
        # Ensure the matched_sections contains data and is not empty
        if 'matched_sections' in result and len(result['matched_sections']) == 0:
            # Try to find more matched sections with a lower threshold for better user experience
            if comparison_type == 'text_text':
                try:
                    # Get file contents
                    with open(filepath1, 'r', encoding='utf-8', errors='ignore') as f:
                        text1 = f.read()
                    with open(filepath2, 'r', encoding='utf-8', errors='ignore') as f:
                        text2 = f.read()
                        
                    # SequenceMatcher and re already imported at the top
                    
                    # Split texts into sentences for more granular comparison
                    sentences1 = re.split(r'(?<=[.!?])\s+', text1)
                    sentences2 = re.split(r'(?<=[.!?])\s+', text2)
                    
                    matcher = SequenceMatcher(None, sentences1, sentences2)
                    matching_blocks = matcher.get_matching_blocks()
                    
                    # Create basic matched sections from raw text
                    basic_matched_sections = []
                    for block in matching_blocks:
                        i, j, size = block
                        if size > 0:
                            # Extract at least 15 characters to show something
                            s1 = ' '.join(sentences1[i:i+size])
                            s2 = ' '.join(sentences2[j:j+size])
                            if len(s1) >= 15:  # Only include substantial matches
                                basic_matched_sections.append({
                                    'file1_text': s1,
                                    'file2_text': s2,
                                    'similarity': result['similarity']  # Use overall similarity
                                })
                    
                    if basic_matched_sections:
                        result_data['matched_sections'] = basic_matched_sections[:5]  # Limit to top 5
                except Exception as e:
                    logging.error(f"Error finding basic matched sections: {str(e)}")
        
        # Store the result data in the session
        session['result'] = json.dumps(result_data)
        
        return redirect(url_for('show_results'))
    
    except Exception as e:
        logging.error(f"Error processing files: {str(e)}")
        flash(f"Error processing files: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/results')
def show_results():
    if 'result' not in session:
        flash('No comparison results available', 'error')
        return redirect(url_for('index'))
    
    result = json.loads(session['result'])
    return render_template('results.html', result=result)

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index')), 413

@app.errorhandler(500)
def internal_server_error(error):
    flash('An unexpected error occurred. Please try again.', 'error')
    return redirect(url_for('index')), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
