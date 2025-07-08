document.addEventListener('DOMContentLoaded', function() {
    // File upload preview
    const fileInputs = document.querySelectorAll('.file-input');
    const uploadAreas = document.querySelectorAll('.upload-area');
    const fileNameDisplays = document.querySelectorAll('.selected-file');
    const plagiarismForm = document.getElementById('plagiarism-form');
    
    // Set up file upload interaction
    fileInputs.forEach((input, index) => {
        input.addEventListener('change', function(e) {
            // Prevent form submission when file input changes
            e.stopPropagation();
            
            if (this.files && this.files[0]) {
                const fileName = this.files[0].name;
                fileNameDisplays[index].textContent = fileName;
                uploadAreas[index].classList.add('has-file');
            }
        });
        
        // Handle click on upload area
        uploadAreas[index].addEventListener('click', function(e) {
            // Only trigger file input click for direct user clicks
            // and not programmatic or event bubbling clicks
            if (e.isTrusted && e.target !== input) {
                e.preventDefault();
                e.stopPropagation();
                fileInputs[index].click();
            }
        });
        
        // Prevent default behavior for drag and drop
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadAreas[index].addEventListener(eventName, preventDefaults, false);
            // Also prevent events on document to avoid browser defaults
            document.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Highlight drop area when dragging over
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadAreas[index].addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadAreas[index].addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            uploadAreas[index].classList.add('drag-over');
        }
        
        function unhighlight() {
            uploadAreas[index].classList.remove('drag-over');
        }
        
        // Handle dropped files
        uploadAreas[index].addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            preventDefaults(e);
            
            // Get the dropped files
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files && files.length > 0) {
                // Set files to the input
                input.files = files;
                
                // Update UI to show the file name
                const fileName = files[0].name;
                fileNameDisplays[index].textContent = fileName;
                uploadAreas[index].classList.add('has-file');
                
                // Dispatch change event so other listeners know the file changed
                const changeEvent = new Event('change', { bubbles: true });
                input.dispatchEvent(changeEvent);
            }
        }
    });
    
    // Comparison type selector
    const comparisonTypeSelector = document.getElementById('comparison-type');
    const programmingLanguageSection = document.getElementById('programming-language-section');
    
    if (comparisonTypeSelector) {
        comparisonTypeSelector.addEventListener('change', function() {
            // Show/hide programming language selector for GitHub repositories comparison
            if (this.value === 'github_github') {
                programmingLanguageSection.classList.remove('d-none');
            } else {
                programmingLanguageSection.classList.add('d-none');
            }
        });
    }
    
    // Form submission
    // plagiarismForm already defined above
    const submitButton = document.getElementById('submit-button');
    const spinner = document.getElementById('spinner');
    
    // Track if the form has been submitted to prevent multiple submissions
    let formSubmitted = false;
    
    if (plagiarismForm) {
        // Prevent default form submission to allow validation
        plagiarismForm.addEventListener('submit', function(e) {
            // If form already submitted, prevent multiple submissions
            if (formSubmitted) {
                e.preventDefault();
                return false;
            }
            
            // Check if files are selected
            const file1 = document.getElementById('file1');
            const file2 = document.getElementById('file2');
            
            if (!file1 || !file1.files || file1.files.length === 0 ||
                !file2 || !file2.files || file2.files.length === 0) {
                e.preventDefault();
                alert('Please select two files for comparison');
                return false;
            }
            
            // Mark as submitted and show loading spinner
            formSubmitted = true;
            
            if (submitButton && spinner) {
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span> Processing...';
            }
            
            // Allow form to submit after validation
            return true;
        });
    }
    
    // Alert auto-dismiss
    const alerts = document.querySelectorAll('.alert-dismissible');
    alerts.forEach(alert => {
        setTimeout(() => {
            const closeButton = new bootstrap.Alert(alert);
            closeButton.close();
        }, 5000);
    });
    
    // Tooltips initialization
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
