// AI Video Detective - Professional JavaScript Application

class VideoDetective {
    constructor() {
        this.currentFile = null;
        this.analysisComplete = false;
        this.isTyping = false;
        this.init();
    }

    init() {
        console.log('🚀 Initializing AI Video Detective Pro...');
        this.setupEventListeners();
        this.setupAutoResize();
        this.checkSessionStatus();
        this.setupPageCleanup();
        this.showDemoVideoPreview();
        // Model selection functionality removed - always using 32B model
        console.log('✅ AI Video Detective Pro initialized successfully!');
    }

    setupEventListeners() {
        // File upload
        const videoFile = document.getElementById('videoFile');
        const uploadArea = document.getElementById('uploadArea');

        if (!videoFile) {
            console.error('❌ videoFile element not found');
            return;
        }

        if (!uploadArea) {
            console.error('❌ uploadArea element not found');
            return;
        }

        videoFile.addEventListener('change', (e) => this.handleFileSelect(e));

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });

        uploadArea.addEventListener('click', (e) => {
            // Don't open file dialog if clicking on other interactive elements
            if (e.target.closest('.upload-buttons') || 
                e.target.closest('.upload-features')) {
                return;
            }
            videoFile.click();
        });

        // Chat functionality
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');

        if (chatInput && sendBtn) {
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
            
            sendBtn.addEventListener('click', () => this.sendMessage());
        } else {
            console.log('ℹ️ Chat elements not found, chat functionality disabled');
        }
        

        
        // Analysis form submission
        const analysisFormElement = document.getElementById('analysisFormElement');
        if (analysisFormElement) {
            analysisFormElement.addEventListener('submit', (e) => {
                e.preventDefault();
                this.handleAnalysisSubmit();
            });
        }
        
        // Reset button
        const resetBtn = document.getElementById('resetBtn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetAnalysis());
        }
    }

    setupAutoResize() {
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.addEventListener('input', () => {
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
            });
        }
    }

    setupPageCleanup() {
        // Clean up old uploads when page loads
        this.cleanupOldUploads();
        
        // Clean up when user navigates away or refreshes
        window.addEventListener('beforeunload', () => {
            this.cleanupOldUploads();
        });
        
        // Clean up when user goes back to upload screen
        window.addEventListener('popstate', () => {
            this.cleanupOldUploads();
        });
    }

    async cleanupOldUploads() {
        try {
            const response = await fetch('/api/cleanup-uploads', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            if (result.success) {
                console.log('🧹 Old uploads cleaned up');
            }
        } catch (error) {
            console.log('Cleanup request failed (normal on page unload)');
        }
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        if (!this.isValidVideoFile(file)) {
            this.showError('Please select a valid video file (MP4, AVI, MOV, WebM, MKV)');
            return;
        }

        if (file.size > 500 * 1024 * 1024) { // 500MB limit
            this.showError('File size must be less than 500MB');
            return;
        }

        this.currentFile = file;
        this.showFileInfo(file);
        this.showVideoPreview(file);
        
        // Show the analysis form after file is selected
        this.showAnalysisForm();
    }

    isValidVideoFile(file) {
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm', 'video/x-matroska'];
        return validTypes.includes(file.type);
    }

    showFileInfo(file, filename = null, fileSize = null) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSizeElement = document.getElementById('fileSize');
        
        // Use provided filename or fallback to file.name
        fileName.textContent = filename || file.name;
        fileSizeElement.textContent = fileSize || this.formatFileSize(file.size);
        fileInfo.style.display = 'block';
        
        console.log('📁 File info displayed:', { name: filename || file.name, size: fileSize || this.formatFileSize(file.size) });
    }

    showAnalysisForm() {
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm) {
            analysisForm.style.display = 'block';
            // Scroll to the analysis form
            analysisForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
            console.log('📋 Analysis form displayed');
        } else {
            console.error('❌ Analysis form element not found');
        }
    }

    async handleAnalysisSubmit() {
        console.log('🚀 Starting comprehensive video analysis...');
        
        // Check if we have a current file
        if (!this.currentFile) {
            this.showError('No video file selected. Please upload a video first.');
            return;
        }
        
        // Hide the analysis form and show progress
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm) {
            analysisForm.style.display = 'none';
        }
        
        // Show progress section
        this.showProgressSection();
        
        // FIRST: Upload the video file
        console.log('📤 Uploading video file before analysis...');
        try {
            await this.uploadFile(this.currentFile);
            console.log('✅ Video uploaded successfully, now starting analysis...');
            
            // SECOND: Start the analysis
            console.log('🚀 About to call startAnalysis() function...');
            this.startAnalysis();
            console.log('🚀 startAnalysis() function called');
        } catch (error) {
            console.error('❌ Upload failed:', error);
            this.showError('Failed to upload video: ' + error.message);
            
            // Show analysis form again on error
            if (analysisForm) {
                analysisForm.style.display = 'block';
            }
        }
    }

    resetAnalysis() {
        // Hide analysis form
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm) {
            analysisForm.style.display = 'none';
        }
        
        // Reset file info
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.style.display = 'none';
        }
        
        // Reset video preview
        const videoPreview = document.getElementById('videoPreview');
        if (videoPreview) {
            videoPreview.style.display = 'none';
        }
        
        // Reset current file
        this.currentFile = null;
        
        console.log('🔄 Analysis reset');
    }

    showProgressSection() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'block';
            progressSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
            console.log('📊 Progress section displayed');
        }
    }

    async startAnalysis() {
        console.log('🚀 Starting comprehensive AI analysis via API...');
        
        try {
            // Show progress section
            this.showProgressSection();
            
            // Verify we have a current file
            if (!this.currentFile) {
                throw new Error('No video file available for analysis');
            }
            
            console.log('🔍 About to call /analyze API...');
            console.log('🔍 Current file:', this.currentFile.name, this.currentFile.size);
            
            // Show progress message
            const progressStatus = document.getElementById('progressStatus');
            if (progressStatus) {
                progressStatus.textContent = 'Starting AI analysis... This may take several minutes for large videos. Please wait patiently.';
            }
            
            // Start analysis with improved polling strategy
            console.log('🚀 Starting analysis with improved polling strategy...');
            
            // Start the analysis (don't wait for completion)
            const startResponse = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    analysis_type: 'comprehensive_analysis',
                    user_focus: 'Analyze this video comprehensively for all important events and observations'
                })
            });
            
            if (!startResponse.ok) {
                throw new Error(`Analysis failed to start: ${startResponse.status}`);
            }
            
            console.log('✅ Analysis started successfully, beginning polling...');
            
            // Update progress message
            if (progressStatus) {
                progressStatus.textContent = 'AI analysis started! Polling for completion... This may take 1-2 minutes.';
            }
            
            // Poll for completion every 3 seconds with better error handling
            let pollCount = 0;
            const maxPolls = 200; // 10 minutes max (200 * 3 seconds)
            
            const pollInterval = setInterval(async () => {
                pollCount++;
                
                try {
                    console.log(`🔍 Polling attempt ${pollCount}/${maxPolls}...`);
                    const statusResponse = await fetch('/api/session/status');
                    
                    if (!statusResponse.ok) {
                        console.warn('⚠️ Status check failed:', statusResponse.status);
                        if (pollCount >= maxPolls) {
                            clearInterval(pollInterval);
                            this.handleAnalysisTimeout();
                        }
                        return;
                    }
                    
                    const statusData = await statusResponse.json();
                    console.log('📊 Status response:', statusData);
                    
                    // Check if analysis is complete
                    if (statusData.analysis_result && statusData.analysis_result.length > 100) {
                        clearInterval(pollInterval);
                        console.log('🎉 Analysis completed!');
                        
                        // Show results
                        this.analysisComplete = true;
                        this.showChatInterface();
                        
                        // Show completion message
                        let completionMessage = '🎯 **Video Analysis Complete!**\n\nYour video has been successfully analyzed. Here\'s what I found:';
                        
                        if (statusData.analysis_result) {
                            completionMessage += '\n\n**Analysis Summary:**\n' + statusData.analysis_result.substring(0, 300) + '...';
                        }
                        
                        completionMessage += '\n\n**Next Steps**: Ask me anything about the video content. I can provide detailed insights about what\'s happening in your video.';
                        
                        // Add completion message with typing effect
                        this.addChatMessageWithTyping('ai', completionMessage);
                        
                        // Hide progress section
                        const progressSection = document.getElementById('progressSection');
                        if (progressSection) {
                            progressSection.style.display = 'none';
                        }
                        
                        // Update progress status
                        if (progressStatus) {
                            progressStatus.textContent = '✅ Analysis completed! Chat interface is ready.';
                        }
                        
                        return;
                    }
                    
                    // Update progress with current time
                    if (progressStatus) {
                        progressStatus.textContent = `AI analysis in progress... (${new Date().toLocaleTimeString()}) - Attempt ${pollCount}/${maxPolls}`;
                    }
                    
                    // Check if we've reached max polls
                    if (pollCount >= maxPolls) {
                        clearInterval(pollInterval);
                        this.handleAnalysisTimeout();
                    }
                    
                } catch (error) {
                    console.error('❌ Error during polling:', error);
                    if (pollCount >= maxPolls) {
                        clearInterval(pollInterval);
                        this.handleAnalysisTimeout();
                    }
                }
            }, 3000); // Poll every 3 seconds instead of 5
            
        } catch (error) {
            console.error('❌ Analysis failed:', error);
            
            const progressStatus = document.getElementById('progressStatus');
            if (progressStatus) {
                progressStatus.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        <h6>❌ Analysis Failed</h6>
                        <p class="mb-2">Error: ${error.message}</p>
                        <button class="btn btn-primary btn-sm" onclick="app.startAnalysis()">
                            <i class="fas fa-redo"></i> Try Again
                        </button>
                    </div>
                `;
            }
            
            // Hide progress section on error
            const progressSection = document.getElementById('progressSection');
            if (progressSection) {
                progressSection.style.display = 'none';
            }
        }
    }
    
    handleAnalysisTimeout() {
        console.log('⏰ Analysis timeout - showing manual check option');
        
        const progressStatus = document.getElementById('progressStatus');
        if (progressStatus) {
            progressStatus.innerHTML = `
                <div class="alert alert-warning" role="alert">
                    <h6>⏰ Analysis taking longer than expected</h6>
                    <p class="mb-2">The analysis is still running in the background. You can:</p>
                    <button class="btn btn-primary btn-sm me-2" onclick="app.checkAnalysisStatus()">
                        <i class="fas fa-sync-alt"></i> Check Status Now
                    </button>
                    <button class="btn btn-outline-primary btn-sm" onclick="app.showChatInterface()">
                        <i class="fas fa-comments"></i> Try Chat Anyway
                    </button>
                </div>
            `;
        }
    }
    
    async checkAnalysisStatus() {
        try {
            console.log('🔍 Manual status check requested...');
            
            const progressStatus = document.getElementById('progressStatus');
            if (progressStatus) {
                progressStatus.textContent = 'Checking analysis status...';
            }
            
            const statusResponse = await fetch('/api/session/status');
            
            if (statusResponse.ok) {
                const statusData = await statusResponse.json();
                
                if (statusData.analysis_result && statusData.analysis_result.length > 100) {
                    console.log('🎉 Manual check: Analysis completed!');
                    this.analysisComplete = true;
                    this.showChatInterface();
                    
                    if (progressStatus) {
                        progressStatus.textContent = '✅ Analysis completed! Chat interface is ready.';
                    }
                    
                    // Hide progress section
                    const progressSection = document.getElementById('progressSection');
                    if (progressSection) {
                        progressSection.style.display = 'none';
                    }
                } else {
                    if (progressStatus) {
                        progressStatus.innerHTML = `
                            <div class="alert alert-info" role="alert">
                                <h6>📊 Analysis Status</h6>
                                <p class="mb-2">Analysis is still in progress. Please wait a bit longer or try again.</p>
                                <button class="btn btn-primary btn-sm" onclick="app.checkAnalysisStatus()">
                                    <i class="fas fa-sync-alt"></i> Check Again
                                </button>
                            </p>
                        </div>
                    `;
                }
            } else {
                throw new Error(`Status check failed: ${statusResponse.status}`);
            }
            
        } catch (error) {
            console.error('❌ Manual status check failed:', error);
            
            const progressStatus = document.getElementById('progressStatus');
            if (progressStatus) {
                progressStatus.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        <h6>❌ Status Check Failed</h6>
                        <p class="mb-2">Error: ${error.message}</p>
                        <button class="btn btn-primary btn-sm" onclick="app.checkAnalysisStatus()">
                            <i class="fas fa-sync-alt"></i> Try Again
                        </button>
                    </div>
                `;
            }
        }
    }



    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showVideoPreview(file) {
        const videoPreview = document.getElementById('videoPreview');
        const previewVideo = document.getElementById('previewVideo');
        
        // Create object URL for video preview
        const videoUrl = URL.createObjectURL(file);
        previewVideo.src = videoUrl;
        
        // Show the preview section
        videoPreview.style.display = 'block';
        
        // Scroll to preview
        videoPreview.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        console.log('🎬 Video preview shown for:', file.name);
    }

    async uploadSelectedVideo() {
        // Hide the preview
        const videoPreview = document.getElementById('videoPreview');
        videoPreview.style.display = 'none';
        
        // Check if this is a demo video preview
        const previewVideo = document.getElementById('previewVideo');
        if (previewVideo && previewVideo.src && previewVideo.src.includes('/demo-video')) {
            // This is a demo video, upload it
            await this.uploadDemoVideo();
        } else if (this.currentFile) {
            // This is a regular uploaded file
            await this.uploadFile(this.currentFile);
        } else {
            // If no file is selected but we're in demo mode, upload demo video
            await this.uploadDemoVideo();
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('video', file);

        try {
            this.showProgress();

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.hideProgress();
                this.showFileInfo(file, result.filename, result.file_size);
                this.showCleanupButton();
                // Don't show analysis form here since it's called from analysis flow
                console.log('✅ Upload successful, ready for analysis');
            } else {
                this.hideProgress();
                this.showError(result.error || 'Upload failed');
            }
        } catch (error) {
            this.hideProgress();
            this.showError('Upload failed: ' + error.message);
        }
    }

    showProgress() {
        const progress = document.getElementById('uploadProgress');
        if (!progress) {
            console.warn('⚠️ Upload progress element not found');
            return;
        }
        
        const progressBar = progress.querySelector('.progress-bar');
        if (!progressBar) {
            console.warn('⚠️ Progress bar element not found');
            return;
        }
        
        progress.style.display = 'block';
        progressBar.style.width = '0%';
        
        // Simulate progress
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 90) {
                clearInterval(interval);
            } else {
                width += Math.random() * 10;
                progressBar.style.width = width + '%';
            }
        }, 200);
    }

    hideProgress() {
        const progress = document.getElementById('uploadProgress');
        if (!progress) {
            console.warn('⚠️ Upload progress element not found');
            return;
        }
        
        const progressBar = progress.querySelector('.progress-bar');
        if (!progressBar) {
            console.warn('⚠️ Progress bar element not found');
            return;
        }
        
        progressBar.style.width = '100%';
        setTimeout(() => {
            progress.style.display = 'none';
        }, 500);
    }

    async analyzeVideo() {
        try {
            console.log('🔍 Starting video analysis...');
            this.showLoadingModal('Analyzing Video');

            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    analysis_type: 'comprehensive_analysis',
                    user_focus: 'Analyze this video comprehensively for all important events and observations'
                })
            });

            // Debug: Log response details before parsing
            console.log('🔍 Response status:', response.status);
            console.log('🔍 Response headers:', response.headers);
            console.log('🔍 Response ok:', response.ok);
            
            // Get the raw response text first
            const responseText = await response.text();
            console.log('🔍 Raw response text:', responseText);
            console.log('🔍 Response text length:', responseText.length);
            console.log('🔍 Response text first 200 chars:', responseText.substring(0, 200));
            
            // Try to parse as JSON
            let result;
            try {
                result = JSON.parse(responseText);
                console.log('📊 Analysis response:', result);
            } catch (jsonError) {
                console.error('❌ JSON parsing failed:', jsonError);
                console.error('❌ Raw response that failed to parse:', responseText);
                throw new Error(`Invalid JSON response from server: ${jsonError.message}. Raw response: ${responseText.substring(0, 500)}`);
            }
            
            this.hideLoadingModal();

            if (result.success) {
                console.log('✅ Analysis successful, showing chat interface...');
                this.analysisComplete = true;
                this.showChatInterface();
                
                // Show analysis completion message with evidence if available
                let completionMessage = '🎯 **Video Analysis Setup Complete!**\n\nYour video has been successfully uploaded and processed. Here\'s what I\'ve prepared:';
                
                // DISABLED: Visual Evidence feature
                // No evidence text will be displayed
                
                completionMessage += '\n\n**Current Status**: Video is ready for AI analysis!\n\n**Next Steps**: Ask me anything about the video content. I can provide:\n- Basic video information and metadata\n- Technical specifications and details\n- Analysis setup guidance\n- Help with video processing questions\n\n**For Full AI Analysis**: The server needs the Qwen2.5-VL-32B model loaded to provide detailed content analysis, object recognition, and behavioral insights.';
                
                // Add completion message with typing effect
                this.addChatMessageWithTyping('ai', completionMessage);
                
                // DISABLED: Evidence display
                // No evidence will be shown
            } else {
                console.error('❌ Analysis failed:', result.error);
                this.showError(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.hideLoadingModal();
            this.showError('Analysis failed: ' + error.message);
        }
    }

    // DISABLED: Evidence display function
    /*
    displayEvidence(evidence, title = 'Visual Evidence') {
        const chatMessages = document.getElementById('chatMessages');
        
        // Create evidence container
        const evidenceContainer = document.createElement('div');
        evidenceContainer.className = 'screenshot-evidence';
        
        // Add header
        const header = document.createElement('h4');
        header.innerHTML = `📸 <strong>${title}</strong>`;
        evidenceContainer.appendChild(header);
        
        // Create evidence grid
        const grid = document.createElement('div');
        grid.className = 'evidence-grid';
        
        evidence.forEach(item => {
            const evidenceItem = document.createElement('div');
            evidenceItem.className = 'evidence-item';
            
            if (item.type === 'video_clip') {
                // Create video element
                const video = document.createElement('video');
                video.src = item.url;
                video.controls = true;
                video.preload = 'metadata';
                
                // Create timestamp label for video
                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.innerHTML = `<strong>${this.formatTimestamp(item.start_time)} - ${this.formatTimestamp(item.end_time)}</strong>`;
                
                evidenceItem.appendChild(video);
                evidenceItem.appendChild(timestamp);
            } else {
                // Create image for screenshot
                const img = document.createElement('img');
                img.src = item.url;
                img.alt = `Screenshot at ${item.timestamp}s`;
                img.loading = 'lazy';
                
                // Add click event to open modal
                img.addEventListener('click', () => {
                    this.openEvidenceModal(item);
                });
                
                // Create timestamp label for screenshot
                const timestamp = document.createElement('div');
                timestamp.className = 'timestamp';
                timestamp.innerHTML = `<strong>${this.formatTimestamp(item.timestamp)}</strong>`;
                
                evidenceItem.appendChild(img);
                evidenceItem.appendChild(timestamp);
            }
            
            grid.appendChild(evidenceItem);
        });
        
        evidenceContainer.appendChild(grid);
        
        // Add to chat messages with animation
        evidenceContainer.style.opacity = '0';
        evidenceContainer.style.transform = 'translateY(20px)';
        chatMessages.appendChild(evidenceContainer);
        
        // Animate in
        setTimeout(() => {
            evidenceContainer.style.transition = 'all 0.3s ease';
            evidenceContainer.style.opacity = '1';
            evidenceContainer.style.transform = 'translateY(0)';
        }, 100);
        
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    */

    openEvidenceModal(evidence) {
        const modal = document.getElementById('evidenceModal');
        const modalImage = document.getElementById('evidenceModalImage');
        const modalInfo = document.getElementById('evidenceModalInfo');
        
        modalImage.src = evidence.url;
        
        if (evidence.type === 'video_clip') {
            modalInfo.textContent = `Timestamp: ${this.formatTimestamp(evidence.start_time)} - ${this.formatTimestamp(evidence.end_time)}`;
        } else {
            modalInfo.textContent = `Timestamp: ${this.formatTimestamp(evidence.timestamp)}`;
        }
        
        modal.style.display = 'flex';
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeEvidenceModal();
            }
        });
        
        // Close modal with Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeEvidenceModal();
            }
        });
    }

    closeEvidenceModal() {
        const modal = document.getElementById('evidenceModal');
        modal.style.display = 'none';
    }

    formatTimestamp(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    showChatInterface() {
        console.log('💬 Showing chat interface...');
        
        try {
            const uploadSection = document.getElementById('uploadSection');
            const chatInterface = document.getElementById('chatInterface');
            
            if (!uploadSection || !chatInterface) {
                console.error('❌ Required elements not found:', { uploadSection: !!uploadSection, chatInterface: !!chatInterface });
                return;
            }
            
            // Hide upload section with animation
            uploadSection.style.transition = 'all 0.3s ease';
            uploadSection.style.opacity = '0';
            uploadSection.style.transform = 'translateY(-20px)';
            
            setTimeout(() => {
                uploadSection.style.display = 'none';
                
                // Show chat interface
                chatInterface.style.display = 'flex';
                chatInterface.style.opacity = '0';
                chatInterface.style.transform = 'translateY(20px)';
                
                setTimeout(() => {
                    chatInterface.style.transition = 'all 0.3s ease';
                    chatInterface.style.opacity = '1';
                    chatInterface.style.transform = 'translateY(0)';
                }, 50);
            }, 300);
            
            // Enable chat input and ensure event listeners are working
            const chatInput = document.getElementById('chatInput');
            const sendBtn = document.getElementById('sendBtn');
            
            if (chatInput && sendBtn) {
                // Remove any existing disabled state
                chatInput.disabled = false;
                sendBtn.disabled = false;
                
                // Ensure event listeners are attached
                this.setupChatEventListeners();
                
                // Focus on input
                chatInput.focus();
                
                console.log('✅ Chat interface elements enabled and focused');
            } else {
                console.error('❌ Chat input elements not found');
            }
            
            console.log('✅ Chat interface should now be visible');
            
        } catch (error) {
            console.error('❌ Error showing chat interface:', error);
        }
    }
    
    setupChatEventListeners() {
        console.log('🔧 Setting up chat event listeners...');
        
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        const minimizeChatBtn = document.getElementById('minimizeChatBtn');
        
        if (!chatInput || !sendBtn) {
            console.error('❌ Chat elements not found for event listeners');
            return;
        }
        
        // Remove existing event listeners to prevent duplicates
        const newChatInput = chatInput.cloneNode(true);
        const newSendBtn = sendBtn.cloneNode(true);
        const newMinimizeChatBtn = minimizeChatBtn ? minimizeChatBtn.cloneNode(true) : null;
        
        chatInput.parentNode.replaceChild(newChatInput, chatInput);
        sendBtn.parentNode.replaceChild(newSendBtn, sendBtn);
        if (newMinimizeChatBtn && minimizeChatBtn) {
            minimizeChatBtn.parentNode.replaceChild(newMinimizeChatBtn, minimizeChatBtn);
        }
        
        // Add new event listeners
        newChatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        newSendBtn.addEventListener('click', () => this.sendMessage());
        
        // Setup minimize button functionality
        if (newMinimizeChatBtn) {
            newMinimizeChatBtn.addEventListener('click', () => this.minimizeChat());
        }
        
        // Setup auto-resize for textarea
        newChatInput.addEventListener('input', () => {
            newChatInput.style.height = 'auto';
            newChatInput.style.height = Math.min(newChatInput.scrollHeight, 120) + 'px';
        });
        
        console.log('✅ Chat event listeners setup complete');
    }
    
    minimizeChat() {
        console.log('📱 Minimizing chat interface...');
        
        const chatInterface = document.getElementById('chatInterface');
        const uploadSection = document.getElementById('uploadSection');
        
        if (chatInterface && uploadSection) {
            // Hide chat interface
            chatInterface.style.display = 'none';
            
            // Show upload section with animation
            uploadSection.style.display = 'block';
            uploadSection.style.opacity = '0';
            uploadSection.style.transform = 'translateY(-20px)';
            
            setTimeout(() => {
                uploadSection.style.transition = 'all 0.3s ease';
                uploadSection.style.opacity = '1';
                uploadSection.style.transform = 'translateY(0)';
            }, 50);
            
            console.log('✅ Chat interface minimized, upload section restored');
        }
    }

    async sendMessage() {
        console.log('💬 sendMessage called');
        
        try {
            const chatInput = document.getElementById('chatInput');
            const sendBtn = document.getElementById('sendBtn');
            
            if (!chatInput || !sendBtn) {
                console.error('❌ Chat elements not found in sendMessage');
                return;
            }
            
            const message = chatInput.value.trim();
            console.log('📝 Message to send:', message);

            if (!message || this.isTyping) {
                console.log('⚠️ Message empty or typing in progress');
                return;
            }

            // Add user message
            this.addChatMessage('user', message);
            chatInput.value = '';
            chatInput.style.height = 'auto';
            
            // Disable input while processing
            chatInput.disabled = true;
            sendBtn.disabled = true;

            // Show typing indicator
            this.showTypingIndicator();

            console.log('🚀 Sending message to /chat endpoint...');
            
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            console.log('📡 Response received:', response.status, response.ok);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('📊 Chat result:', result);

            // Hide typing indicator
            this.hideTypingIndicator();

            if (result.success) {
                // Add AI message with typing effect
                this.addChatMessageWithTyping('ai', result.response);
                
                // Display additional evidence if available (after message appears)
                if (result.additional_screenshots && result.additional_screenshots.length > 0) {
                    setTimeout(() => {
                        this.displayEvidence(result.additional_screenshots, 'Additional Evidence');
                    }, 800); // Wait for fade-in effect to complete + buffer
                }
            } else {
                console.error('❌ Chat API returned error:', result.error);
                this.addChatMessage('ai', `Sorry, I encountered an error: ${result.error || 'Unknown error'}. Please try again.`);
            }
        } catch (error) {
            console.error('❌ Error in sendMessage:', error);
            this.hideTypingIndicator();
            this.addChatMessage('ai', `Sorry, I encountered an error: ${error.message}. Please try again.`);
        } finally {
            // Re-enable input
            const chatInput = document.getElementById('chatInput');
            const sendBtn = document.getElementById('sendBtn');
            
            if (chatInput && sendBtn) {
                chatInput.disabled = false;
                sendBtn.disabled = false;
                chatInput.focus();
            }
        }
    }

    showTypingIndicator() {
        this.isTyping = true;
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.style.display = 'block';
        
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('typingIndicator');
        typingIndicator.style.display = 'none';
    }

    addChatMessage(type, message) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;

        const icon = type === 'user' ? 'fas fa-user' : 'fas fa-robot';
        
        // Enhanced message formatting for AI responses
        let formattedMessage = message;
        if (type === 'ai') {
            formattedMessage = this.formatAIResponse(message);
        }

        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-avatar">
                <i class="${icon}"></i>
                </div>
                <div class="message-text"></div>
            </div>
        `;

        // Set the formatted content safely
        const messageTextElement = messageDiv.querySelector('.message-text');
        messageTextElement.innerHTML = formattedMessage;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    addChatMessageWithTyping(type, message) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}-message`;

        const icon = type === 'user' ? 'fas fa-user' : 'fas fa-robot';
        
        // Enhanced message formatting for AI responses
        let formattedMessage = this.formatAIResponse(message);

        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-avatar">
                    <i class="${icon}"></i>
                </div>
                <div class="message-text"></div>
                </div>
            `;

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Show message with fade-in effect
        const messageText = messageDiv.querySelector('.message-text');
        this.showMessageWithEffect(messageText, formattedMessage);
    }

    showMessageWithEffect(element, text) {
        try {
            // Start with opacity 0 and add fade-in effect
            element.style.opacity = '0';
            element.style.transform = 'translateY(10px)';
            element.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
            
            // Set the content immediately
            element.innerHTML = text;
            
            // Trigger the fade-in animation
            setTimeout(() => {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }, 50);
        } catch (error) {
            console.error('❌ Error in showMessageWithEffect:', error);
            // Fallback: just set the content without animation
            if (element) {
                element.innerHTML = text;
            }
        }
    }
    
    showTypingIndicator() {
        try {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.style.display = 'block';
                this.isTyping = true;
            }
        } catch (error) {
            console.error('❌ Error showing typing indicator:', error);
        }
    }
    
    hideTypingIndicator() {
        try {
            const typingIndicator = document.getElementById('typingIndicator');
            if (typingIndicator) {
                typingIndicator.style.display = 'none';
                this.isTyping = false;
            }
        } catch (error) {
            console.error('❌ Error hiding typing indicator:', error);
        }
    }

    formatAIResponse(message) {
        // Convert markdown-style formatting to HTML
        return message
            // Bold text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic text
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Headers
            .replace(/^### (.*$)/gim, '<h4 style="margin: 15px 0 8px 0; color: #1a1a1a; font-weight: 600;">$1</h4>')
            .replace(/^## (.*$)/gim, '<h3 style="margin: 20px 0 10px 0; color: #1a1a1a; font-weight: 600;">$1</h3>')
            .replace(/^# (.*$)/gim, '<h2 style="margin: 25px 0 15px 0; color: #1a1a1a; font-weight: 600;">$1</h2>')
            // Bullet points
            .replace(/^\* (.*$)/gim, '<li>$1</li>')
            .replace(/^- (.*$)/gim, '<li>$1</li>')
            // Numbered lists
            .replace(/^\d+\. (.*$)/gim, '<li>$1</li>')
            // Wrap lists in ul/ol tags
            .replace(/(<li>.*<\/li>)/gs, '<ul style="margin: 12px 0; padding-left: 24px;">$1</ul>')
            // Line breaks
            .replace(/\n/g, '<br>')
            // Timestamps with special styling
            .replace(/(\d{2}:\d{2}-\d{2}:\d{2})/g, '<span style="background: #f3f4f6; padding: 4px 8px; border-radius: 6px; font-family: monospace; font-weight: 600; color: #374151; border: 1px solid #e5e7eb;">$1</span>')
            // Single timestamps
            .replace(/(\d{2}:\d{2})/g, '<span style="background: #f3f4f6; padding: 4px 8px; border-radius: 6px; font-family: monospace; color: #374151;">$1</span>');
    }

    showLoadingModal(message) {
        const modal = document.getElementById('loadingModal');
        const loadingMessage = document.getElementById('loadingMessage');
        
        loadingMessage.textContent = message;
        modal.style.display = 'flex';
        
        // Animate loading steps
        this.animateLoadingSteps();
    }

    animateLoadingSteps() {
        const steps = document.querySelectorAll('.loading-steps .step');
        let currentStep = 0;
        
        const interval = setInterval(() => {
            steps.forEach((step, index) => {
                if (index <= currentStep) {
                    step.classList.add('active');
                } else {
                    step.classList.remove('active');
                }
            });
            
            currentStep++;
            if (currentStep >= steps.length) {
                clearInterval(interval);
            }
        }, 1000);
    }

    hideLoadingModal() {
        const modal = document.getElementById('loadingModal');
        modal.style.display = 'none';
    }

    showError(message) {
        // Create a professional error notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #ef4444;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 3000;
            font-weight: 500;
            max-width: 300px;
            animation: slideInRight 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 5000);
    }

    async cleanupSession() {
        try {
            const response = await fetch('/api/session/cleanup', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const result = await response.json();
            
            if (result.success) {
                this.resetUpload();
                this.hideCleanupButton();
                this.showSuccess('Session cleaned up successfully! All files and data have been removed.');
            } else {
                this.showError('Failed to cleanup session: ' + result.error);
            }
        } catch (error) {
            console.error('Cleanup error:', error);
            this.showError('Error cleaning up session: ' + error.message);
        }
    }

    showSuccess(message) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            z-index: 3000;
            font-weight: 500;
            max-width: 300px;
            animation: slideInRight 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
    }

    async checkSessionStatus() {
        try {
            const response = await fetch('/api/session/status');
            const result = await response.json();
            
            if (result.active) {
                if (result.video_uploaded || result.evidence_count > 0) {
                    this.showCleanupButton();
                } else {
                    this.hideCleanupButton();
                }
                console.log('📊 Session status:', result);
            } else {
                this.hideCleanupButton();
            }
        } catch (error) {
            console.error('Session status check error:', error);
        }
    }

    showCleanupButton() {
        const floatingCleanup = document.getElementById('floatingCleanup');
        if (floatingCleanup) {
            floatingCleanup.style.display = 'block';
        }
    }

    hideCleanupButton() {
        const floatingCleanup = document.getElementById('floatingCleanup');
        if (floatingCleanup) {
            floatingCleanup.style.display = 'none';
        }
    }



    showDemoVideoPreview() {
        try {
            console.log('🎬 Showing demo video preview...');
            
            // Check if demo video route exists
            fetch('/demo-video', { method: 'HEAD' })
                .then(response => {
                    if (response.ok) {
                        // Demo video route exists, show preview
                        const videoPreview = document.getElementById('videoPreview');
                        const previewVideo = document.getElementById('previewVideo');
                        
                        if (videoPreview && previewVideo) {
                            // Set the demo video source
                            previewVideo.src = '/demo-video';
                            
                            // Show the preview section
                            videoPreview.style.display = 'block';
                            
                            console.log('🎬 Demo video preview shown');
                        }
                    } else {
                        console.log('ℹ️ Demo video route not available, skipping preview');
                    }
                })
                .catch(error => {
                    console.log('ℹ️ Demo video route not available, skipping preview:', error.message);
                });
        } catch (error) {
            console.error('Failed to show demo video preview:', error);
        }
    }

    async uploadDemoVideo() {
        try {
            console.log('🎬 Uploading demo video...');
            this.showProgress();
            
            // Upload demo video (no actual file, just trigger the server to use default)
            const formData = new FormData();
            // Don't append any file - this will trigger the server to use default video
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.hideProgress();
                this.showCleanupButton();
                
                // Create a mock file object for the demo video and show file info
                const demoFile = {
                    name: 'BMW M4 - Ultimate Racetrack - BMW Canada (720p, h264).mp4',
                    size: 0,
                    type: 'video/mp4'
                };
                this.showFileInfo(demoFile);
                
                // Update file info with actual data from server
                if (result.filename) {
                    const fileInfo = document.getElementById('fileInfo');
                    const fileName = fileInfo.querySelector('.file-name');
                    if (fileName) {
                        fileName.textContent = result.filename;
                    }
                }
                
                // Show success message
                this.showSuccess('Demo video loaded successfully! 🎬');
                
                // Start analysis
                this.analyzeVideo();
            } else {
                this.hideProgress();
                this.showError(result.error || 'Failed to load demo video');
            }
        } catch (error) {
            this.hideProgress();
            this.showError('Failed to load demo video: ' + error.message);
        }
    }

    resetUpload() {
        // Clear current file
        const videoFile = document.getElementById('videoFile');
        videoFile.value = '';
        
        // Hide file info
        const fileInfo = document.getElementById('fileInfo');
        fileInfo.style.display = 'none';
        
        // Hide video preview and clean up object URL
        const videoPreview = document.getElementById('videoPreview');
        const previewVideo = document.getElementById('previewVideo');
        if (videoPreview) {
            videoPreview.style.display = 'none';
        }
        if (previewVideo && previewVideo.src) {
            if (previewVideo.src.includes('/demo-video')) {
                // For demo video, just clear the src
                previewVideo.src = '';
            } else {
                // For uploaded videos, revoke the object URL
                URL.revokeObjectURL(previewVideo.src);
                previewVideo.src = '';
            }
        }
        
        // Clear current file reference
        this.currentFile = null;
        
        // Show upload section with animation
        const uploadSection = document.getElementById('uploadSection');
        const chatInterface = document.getElementById('chatInterface');
        
        chatInterface.style.transition = 'all 0.3s ease';
        chatInterface.style.opacity = '0';
        chatInterface.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            chatInterface.style.display = 'none';
            
            uploadSection.style.display = 'block';
            uploadSection.style.opacity = '0';
            uploadSection.style.transform = 'translateY(-20px)';
            
            setTimeout(() => {
                uploadSection.style.transition = 'all 0.3s ease';
                uploadSection.style.opacity = '1';
                uploadSection.style.transform = 'translateY(0)';
            }, 50);
        }, 300);
        
        // Clear chat messages
        const chatMessages = document.getElementById('chatMessages');
        chatMessages.innerHTML = '';
        
        // Disable chat input
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        chatInput.disabled = true;
        sendBtn.disabled = true;
        chatInput.style.height = 'auto';
        
        // Reset analysis state
        this.analysisComplete = false;
        
        // Clean up old uploads when returning to home screen
        this.cleanupOldUploads();
        
        console.log('🔄 Upload interface reset');
    }
    
    // Model selection functionality removed - always using 32B model
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Initialize the application
window.videoDetective = new VideoDetective();
window.closeEvidenceModal = () => window.videoDetective.closeEvidenceModal();
window.cleanupSession = () => window.videoDetective.cleanupSession();
window.uploadSelectedVideo = () => window.videoDetective.uploadSelectedVideo();

console.log('🚀 AI Video Detective Pro is ready!');