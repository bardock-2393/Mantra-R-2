// Video Analysis App - Enhanced with Progress Tracking and Timeout Handling
// Prevents Cloudflare 524 errors by implementing async analysis flow

class VideoAnalyzer {
    constructor() {
        this.currentSession = null;
        this.progressInterval = null;
        this.analysisTimeout = null;
        this.maxAnalysisTime = 300000; // 5 minutes in milliseconds
        this.progressCheckInterval = 2000; // Check progress every 2 seconds
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // File upload handling
        const fileInput = document.getElementById('videoFile');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        // Analysis form handling
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm) {
            analysisForm.addEventListener('submit', (e) => this.handleAnalysisSubmit(e));
        }

        // Analysis type selection
        const analysisTypeSelect = document.getElementById('analysisType');
        if (analysisTypeSelect) {
            analysisTypeSelect.addEventListener('change', (e) => this.handleAnalysisTypeChange(e));
        }

        // Focus input handling
        const focusInput = document.getElementById('userFocus');
        if (focusInput) {
            focusInput.addEventListener('input', (e) => this.handleFocusInput(e));
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.displayFileInfo(file);
            this.enableAnalysisForm();
        }
    }

    displayFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            fileInfo.innerHTML = `
                <div class="alert alert-info">
                    <strong>Selected File:</strong> ${file.name}<br>
                    <strong>Size:</strong> ${sizeMB} MB<br>
                    <strong>Type:</strong> ${file.type || 'Unknown'}
                </div>
            `;
        }
    }

    enableAnalysisForm() {
        const analysisForm = document.getElementById('analysisForm');
        const submitBtn = document.getElementById('submitAnalysis');
        if (analysisForm && submitBtn) {
            analysisForm.style.display = 'block';
            submitBtn.disabled = false;
        }
    }

    handleAnalysisTypeChange(event) {
        const selectedType = event.target.value;
        const focusInput = document.getElementById('userFocus');
        
        if (focusInput) {
            // Set default focus based on analysis type
            const defaultFocuses = {
                'general': 'Analyze this video comprehensively',
                'behavioral': 'Focus on behavior patterns and actions',
                'technical': 'Provide technical analysis of content',
                'narrative': 'Analyze storytelling and narrative structure',
                'forensic': 'Conduct forensic analysis for evidence',
                'commercial': 'Analyze from marketing perspective',
                'comprehensive_analysis': 'Provide comprehensive multi-dimensional analysis',
                'safety_investigation': 'Conduct thorough safety analysis',
                'creative_review': 'Provide creative and aesthetic analysis'
            };
            
            focusInput.value = defaultFocuses[selectedType] || 'Analyze this video comprehensively';
        }
    }

    handleFocusInput(event) {
        // Real-time character count and validation
        const maxLength = 500;
        const currentLength = event.target.value.length;
        const charCount = document.getElementById('charCount');
        
        if (charCount) {
            charCount.textContent = `${currentLength}/${maxLength}`;
            
            if (currentLength > maxLength * 0.9) {
                charCount.style.color = currentLength > maxLength ? 'red' : 'orange';
            } else {
                charCount.style.color = 'inherit';
            }
        }
    }

    async handleAnalysisSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(event.target);
        const analysisType = formData.get('analysisType');
        const userFocus = formData.get('userFocus');
        
        // Validate inputs
        if (!analysisType || !userFocus.trim()) {
            this.showError('Please select an analysis type and provide a focus area.');
            return;
        }
        
        // Check if file is selected
        const fileInput = document.getElementById('videoFile');
        if (!fileInput.files[0]) {
            this.showError('Please select a video file first.');
            return;
        }
        
        // Start analysis process
        await this.startAnalysis(analysisType, userFocus);
    }

    async startAnalysis(analysisType, userFocus) {
        try {
            // Show loading state
            this.showLoadingState();
            
            // First, upload the video if not already uploaded
            const uploadResult = await this.uploadVideo();
            if (!uploadResult.success) {
                throw new Error(uploadResult.error || 'Video upload failed');
            }
            
            // Start analysis with progress tracking
            const analysisResult = await this.startVideoAnalysis(analysisType, userFocus);
            
            if (analysisResult.success) {
                // Analysis started successfully, begin progress tracking
                this.currentSession = analysisResult.session_id;
                this.startProgressTracking();
                
                // Show progress UI
                this.showProgressUI();
                
                // Set overall timeout
                this.analysisTimeout = setTimeout(() => {
                    this.handleAnalysisTimeout();
                }, this.maxAnalysisTime);
                
            } else {
                throw new Error(analysisResult.error || 'Failed to start analysis');
            }
            
        } catch (error) {
            console.error('Analysis start error:', error);
            this.showError(`Failed to start analysis: ${error.message}`);
            this.hideLoadingState();
        }
    }

    async uploadVideo() {
        const fileInput = document.getElementById('videoFile');
        const file = fileInput.files[0];
        
        if (!file) {
            throw new Error('No video file selected');
        }
        
        const formData = new FormData();
        formData.append('video', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
            }
            
            const result = await response.json();
            return result;
            
        } catch (error) {
            console.error('Upload error:', error);
            throw new Error(`Video upload failed: ${error.message}`);
        }
    }

    async startVideoAnalysis(analysisType, userFocus) {
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    analysis_type: analysisType,
                    user_focus: userFocus
                })
            });
            
            if (!response.ok) {
                throw new Error(`Analysis request failed: ${response.status} ${response.statusText}`);
            }
            
            const result = await response.json();
            return result;
            
        } catch (error) {
            console.error('Analysis request error:', error);
            throw new Error(`Failed to start analysis: ${error.message}`);
        }
    }

    startProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
        }
        
        this.progressInterval = setInterval(async () => {
            await this.checkProgress();
        }, this.progressCheckInterval);
    }

    async checkProgress() {
        if (!this.currentSession) return;
        
        try {
            const response = await fetch(`/api/analysis-progress/${this.currentSession}`);
            
            if (!response.ok) {
                throw new Error(`Progress check failed: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.success) {
                // Update progress UI
                this.updateProgressUI(result.progress);
                
                // Check if analysis is complete
                if (result.completed) {
                    this.handleAnalysisComplete(result);
                }
            } else {
                throw new Error(result.error || 'Progress check failed');
            }
            
        } catch (error) {
            console.error('Progress check error:', error);
            this.showError(`Progress check failed: ${error.message}`);
        }
    }

    updateProgressUI(progress) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const progressStatus = document.getElementById('progressStatus');
        
        if (progressBar && progressText && progressStatus) {
            progressBar.value = progress.progress;
            progressText.textContent = `${progress.progress}%`;
            progressStatus.textContent = progress.message;
            
            // Update progress bar color based on status
            if (progress.status === 'error' || progress.status === 'timeout') {
                progressBar.className = 'progress-bar bg-danger';
            } else if (progress.status === 'completed' || progress.status === 'completed_fallback') {
                progressBar.className = 'progress-bar bg-success';
            } else {
                progressBar.className = 'progress-bar bg-primary';
            }
        }
    }

    handleAnalysisComplete(result) {
        // Clear intervals and timeouts
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        if (this.analysisTimeout) {
            clearTimeout(this.analysisTimeout);
            this.analysisTimeout = null;
        }
        
        // Hide progress UI
        this.hideProgressUI();
        
        // Show results
        if (result.results && result.results.success) {
            this.displayAnalysisResults(result.results);
        } else {
            this.showError(result.error || 'Analysis failed');
        }
        
        // Reset state
        this.currentSession = null;
        this.hideLoadingState();
    }

    handleAnalysisTimeout() {
        console.warn('Analysis timed out');
        
        // Clear progress tracking
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        
        // Hide progress UI
        this.hideProgressUI();
        
        // Show timeout error
        this.showError('Analysis timed out. The video may be too long or complex. Please try with a shorter video or contact support.');
        
        // Reset state
        this.currentSession = null;
        this.hideLoadingState();
    }

    showProgressUI() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
        
        // Hide other sections
        this.hideResultsSection();
        this.hideErrorSection();
    }

    hideProgressUI() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
    }

    displayAnalysisResults(results) {
        const resultsSection = document.getElementById('resultsSection');
        if (!resultsSection) return;
        
        // Display analysis text
        const analysisText = document.getElementById('analysisText');
        if (analysisText) {
            analysisText.innerHTML = this.formatAnalysisText(results.analysis);
        }
        
        // Display timestamps
        const timestampsList = document.getElementById('timestampsList');
        if (timestampsList && results.timestamps) {
            timestampsList.innerHTML = results.timestamps.map(timestamp => 
                `<li class="list-group-item">${timestamp}</li>`
            ).join('');
        }
        
        // Display video duration
        const durationInfo = document.getElementById('durationInfo');
        if (durationInfo && results.video_duration) {
            const minutes = Math.floor(results.video_duration / 60);
            const seconds = Math.floor(results.video_duration % 60);
            durationInfo.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Hide other sections
        this.hideProgressUI();
        this.hideErrorSection();
    }

    formatAnalysisText(text) {
        // Convert markdown-like formatting to HTML
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/^/, '<p>')
            .replace(/$/, '</p>');
    }

    showLoadingState() {
        const submitBtn = document.getElementById('submitAnalysis');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Starting...';
        }
    }

    hideLoadingState() {
        const submitBtn = document.getElementById('submitAnalysis');
        if (submitBtn) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = 'Analyze Video';
        }
    }

    showError(message) {
        const errorSection = document.getElementById('errorSection');
        const errorText = document.getElementById('errorText');
        
        if (errorSection && errorText) {
            errorText.textContent = message;
            errorSection.style.display = 'block';
            
            // Hide other sections
            this.hideProgressUI();
            this.hideResultsSection();
        }
        
        console.error('Error:', message);
    }

    hideErrorSection() {
        const errorSection = document.getElementById('errorSection');
        if (errorSection) {
            errorSection.style.display = 'none';
        }
    }

    hideResultsSection() {
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'none';
        }
    }

    // Utility method to reset the form
    resetForm() {
        const form = document.getElementById('analysisForm');
        if (form) {
            form.reset();
        }
        
        // Clear file selection
        const fileInput = document.getElementById('videoFile');
        if (fileInput) {
            fileInput.value = '';
        }
        
        // Hide all sections
        this.hideProgressUI();
        this.hideResultsSection();
        this.hideErrorSection();
        
        // Reset file info
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.innerHTML = '';
        }
        
        // Hide analysis form
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm) {
            analysisForm.style.display = 'none';
        }
        
        // Reset state
        this.currentSession = null;
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
        if (this.analysisTimeout) {
            clearTimeout(this.analysisTimeout);
            this.analysisTimeout = null;
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.videoAnalyzer = new VideoAnalyzer();
    
    // Add reset button functionality
    const resetBtn = document.getElementById('resetBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', () => {
            window.videoAnalyzer.resetForm();
        });
    }
    
    // Add character count display
    const focusInput = document.getElementById('userFocus');
    if (focusInput) {
        const charCount = document.createElement('small');
        charCount.id = 'charCount';
        charCount.className = 'text-muted';
        charCount.textContent = '0/500';
        focusInput.parentNode.appendChild(charCount);
    }
});

// Export for global access
window.VideoAnalyzer = VideoAnalyzer;