// AI Video Detective - Professional JavaScript Application

class VideoDetective {
    constructor() {
        this.currentFile = null;
        this.analysisComplete = false;
        this.isTyping = false;
        this.fileUploaded = false; // Add missing property
        this.init();
    }

    init() {
        console.log('🚀 Initializing AI Video Detective Pro...');
        console.log('🔍 Checking if functions are available...');
        console.log('startAnalysis available:', typeof this.startAnalysis);
        console.log('uploadDemoVideo available:', typeof this.uploadDemoVideo);
        console.log('testUpload available:', typeof this.testUpload);
        console.log('resetUpload available:', typeof this.resetUpload);
        
        this.setupEventListeners();
        this.setupAutoResize();
        this.checkSessionStatus();
        this.setupPageCleanup();
        this.showDemoVideoPreview();
        this.initializeModelSelection();
        
        // Debug: Check if upload button is visible
        setTimeout(() => {
            this.debugUploadButton();
        }, 1000);
        
        console.log('✅ AI Video Detective Pro initialized successfully!');
    }

    setupEventListeners() {
        console.log('🔧 Setting up event listeners...');
        
        // File upload
        const videoFile = document.getElementById('videoFile');
        const uploadArea = document.getElementById('uploadArea');
        
        console.log('📁 Video file input found:', videoFile);
        console.log('📤 Upload area found:', uploadArea);

        if (videoFile) {
            videoFile.addEventListener('change', (e) => this.handleFileSelect(e));
            console.log('✅ File change event listener added');
        } else {
            console.error('❌ Video file input not found!');
        }

        if (uploadArea) {
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
                // Don't open file dialog if clicking on model selection or other interactive elements
                if (e.target.closest('.model-selection') || 
                    e.target.closest('.upload-buttons') || 
                    e.target.closest('.upload-features')) {
                    return;
                }
                console.log('🖱️ Upload area clicked, opening file dialog...');
                videoFile.click();
            });
            
            console.log('✅ Upload area event listeners added');
        } else {
            console.error('❌ Upload area not found!');
        }

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
            console.log('✅ Chat event listeners added');
        } else {
            console.log('⚠️ Chat elements not found (normal on initial load)');
        }
        
        console.log('🔧 Event listeners setup complete');
    }

    setupAutoResize() {
        const chatInput = document.getElementById('chatInput');
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });
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
        console.log('📂 File selected:', file ? file.name : 'None');
        if (file) {
            this.handleFile(file);
        }
    }

    handleFile(file) {
        console.log('🎬 Handling file:', file.name, file.size, file.type);
        
        if (!this.isValidVideoFile(file)) {
            this.showError('Please select a valid video file (MP4, AVI, MOV, WebM, MKV)');
            return;
        }

        if (file.size > 500 * 1024 * 1024) { // 500MB limit
            this.showError('File size must be less than 500MB');
            return;
        }

        console.log('✅ File validation passed, setting up UI...');
        this.currentFile = file;
        this.fileUploaded = false; // Reset upload status for new file
        this.showFileInfo(file);
        this.showVideoPreview(file);
        this.showAnalysisForm();
        this.updateAnalysisButtonState();
    }

    isValidVideoFile(file) {
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm', 'video/x-matroska'];
        return validTypes.includes(file.type);
    }

    showFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        
        console.log('📁 Showing file info:', file.name, file.size);
        
        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = this.formatFileSize(file.size);
        
        if (fileInfo) {
            fileInfo.style.display = 'block';
        }
        
        // Don't mark file as uploaded here - it needs to go through actual upload process
        console.log('📁 File info displayed, waiting for upload...');
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

    showAnalysisForm() {
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm && this.currentFile) {
            analysisForm.style.display = 'block';
            analysisForm.scrollIntoView({ behavior: 'smooth', block: 'center' });
            this.updateAnalysisButtonState();
        }
    }

    updateAnalysisButtonState() {
        const submitAnalysisBtn = document.getElementById('submitAnalysis');
        if (submitAnalysisBtn) {
            console.log('🔍 Updating analysis button state:');
            console.log('  - currentFile:', this.currentFile ? this.currentFile.name : 'None');
            console.log('  - fileUploaded:', this.fileUploaded);
            
            if (this.currentFile) {
                submitAnalysisBtn.disabled = false;
                submitAnalysisBtn.innerHTML = '<i class="fas fa-play"></i> Analyze Video';
                submitAnalysisBtn.onclick = () => { this.uploadAndAnalyze(); return false; };
                console.log('✅ Button enabled: Analyze Video');
            } else {
                submitAnalysisBtn.disabled = true;
                submitAnalysisBtn.innerHTML = '<i class="fas fa-clock"></i> Please Upload Video First';
                submitAnalysisBtn.onclick = null;
                console.log('❌ Button disabled: Please Upload Video First');
            }
        } else {
            console.log('⚠️ Analysis button not found');
        }
    }

    hideAnalysisForm() {
        const analysisForm = document.getElementById('analysisForm');
        if (analysisForm) {
            analysisForm.style.display = 'none';
        }
    }

    async startAnalysis() {
        console.log('🚀 startAnalysis function called!');
        console.log('📁 Current file:', this.currentFile);
        console.log('📤 File uploaded flag:', this.fileUploaded);
        
        if (!this.currentFile) {
            console.log('❌ No current file, showing error');
            this.showError('Please select a video file first');
            return;
        }

        // Check if we have a valid file before proceeding
        if (!this.currentFile.name || this.currentFile.size === 0) {
            console.log('❌ Invalid file, showing error');
            this.showError('Invalid video file. Please select a valid video file.');
            return;
        }

        try {
            console.log('🔍 Starting video analysis...');
            
            // Show progress section instead of loading modal
            this.showProgressSection();
            this.updateProgress(10, 'Initializing video analysis...');

            // First upload the file if it hasn't been uploaded yet
            if (!this.fileUploaded) {
                console.log('📤 File not uploaded yet, uploading first...');
                this.updateProgress(20, 'Uploading video file...');
                await this.uploadFile(this.currentFile);
                this.fileUploaded = true;
            }

            this.updateProgress(30, 'Starting AI analysis...');
            
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
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
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
            
            this.updateProgress(90, 'Analysis completed, processing results...');
            this.hideProgressSection();

            if (result.success) {
                console.log('✅ Analysis successful, showing chat interface...');
                this.analysisComplete = true;
                this.showChatInterface();
                
                // Show analysis completion message with evidence if available
                let completionMessage = '🎯 **Video Analysis Complete!**\n\nI\'ve thoroughly analyzed your video and captured key insights. Here\'s what I found:';
                
                if (result.evidence && result.evidence.length > 0) {
                    const screenshotCount = result.evidence.filter(e => e.type === 'screenshot').length;
                    const videoCount = result.evidence.filter(e => e.type === 'video_clip').length;
                    
                    let evidenceText = '';
                    if (screenshotCount > 0 && videoCount > 0) {
                        evidenceText = `📸 **Visual Evidence**: I've captured ${screenshotCount} screenshots and ${videoCount} video clips at key moments.`;
                    } else if (screenshotCount > 0) {
                        evidenceText = `📸 **Visual Evidence**: I've captured ${screenshotCount} screenshots at key timestamps.`;
                    } else if (videoCount > 0) {
                        evidenceText = `🎥 **Visual Evidence**: I've captured ${videoCount} video clips at key moments.`;
                    }
                    completionMessage += `\n\n${evidenceText}`;
                }
                
                completionMessage += '\n\n**Ask me anything about the video content!** I can provide detailed insights about specific moments, events, objects, or any aspect you\'re interested in.';
                
                // Add completion message with typing effect
                this.addChatMessageWithTyping('ai', completionMessage);
                
                // Display evidence if available (after message appears)
                if (result.evidence && result.evidence.length > 0) {
                    setTimeout(() => {
                        this.displayEvidence(result.evidence);
                    }, 800); // Wait for fade-in effect to complete + buffer
                }
            } else {
                console.error('❌ Analysis failed:', result.error);
                let errorMessage = result.error || 'Analysis failed';
                
                // Provide more specific error messages
                if (errorMessage.includes('No video uploaded')) {
                    errorMessage = 'No video found. Please upload a video first, then try analysis again.';
                } else if (errorMessage.includes('session')) {
                    errorMessage = 'Session error. Please refresh the page and try again.';
                }
                
                this.showError(errorMessage);
            }
        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.hideProgressSection();
            
            let errorMessage = error.message;
            
            // Provide more specific error messages
            if (errorMessage.includes('JSON')) {
                errorMessage = 'Server response error. Please try again or contact support.';
            } else if (errorMessage.includes('fetch')) {
                errorMessage = 'Network error. Please check your connection and try again.';
            }
            
            this.showError('Analysis failed: ' + errorMessage);
        }
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
        console.log('📤 Starting file upload:', file.name, file.size, file.type);
        
        const formData = new FormData();
        formData.append('video', file);
        
        console.log('📦 FormData created with file');

        try {
            console.log('🔄 Showing progress bar...');
            this.showProgress();

            console.log('📡 Sending upload request to /upload...');
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            console.log('📥 Upload response received:', response.status, response.statusText);
            console.log('📋 Response headers:', response.headers);

            const result = await response.json();
            console.log('📊 Upload result:', result);

            if (result.success) {
                console.log('✅ Upload successful!');
                this.hideProgress();
                this.showFileInfo(file, result.filename, result.file_size);
                this.showCleanupButton();
                this.fileUploaded = true; // Mark file as uploaded
                this.updateAnalysisButtonState(); // Update button state
                // Don't auto-analyze, let user click the analysis button
                this.showSuccess('Video uploaded successfully! Click "Start Analysis" to begin analysis.');
            } else {
                console.log('❌ Upload failed:', result.error);
                this.hideProgress();
                this.showError(result.error || 'Upload failed');
            }
        } catch (error) {
            console.log('💥 Upload error caught:', error);
            this.hideProgress();
            this.showError('Upload failed: ' + error.message);
        }
    }

    showProgress() {
        const progress = document.getElementById('uploadProgress');
        const progressBar = progress.querySelector('.progress-bar');
        
        if (progress && progressBar) {
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
    }

    hideProgress() {
        const progress = document.getElementById('uploadProgress');
        const progressBar = progress.querySelector('.progress-bar');
        
        if (progress && progressBar) {
            progressBar.style.width = '100%';
            setTimeout(() => {
                progress.style.display = 'none';
            }, 500);
        }
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
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
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
                let completionMessage = '🎯 **Video Analysis Complete!**\n\nI\'ve thoroughly analyzed your video and captured key insights. Here\'s what I found:';
                
                if (result.evidence && result.evidence.length > 0) {
                    const screenshotCount = result.evidence.filter(e => e.type === 'screenshot').length;
                    const videoCount = result.evidence.filter(e => e.type === 'video_clip').length;
                    
                    let evidenceText = '';
                    if (screenshotCount > 0 && videoCount > 0) {
                        evidenceText = `📸 **Visual Evidence**: I've captured ${screenshotCount} screenshots and ${videoCount} video clips at key moments.`;
                    } else if (screenshotCount > 0) {
                        evidenceText = `📸 **Visual Evidence**: I've captured ${screenshotCount} screenshots at key timestamps.`;
                    } else if (videoCount > 0) {
                        evidenceText = `🎥 **Visual Evidence**: I've captured ${videoCount} video clips at key moments.`;
                    }
                    completionMessage += `\n\n${evidenceText}`;
                }
                
                completionMessage += '\n\n**Ask me anything about the video content!** I can provide detailed insights about specific moments, events, objects, or any aspect you\'re interested in.';
                
                // Add completion message with typing effect
                this.addChatMessageWithTyping('ai', completionMessage);
                
                // Display evidence if available (after message appears)
                if (result.evidence && result.evidence.length > 0) {
                    setTimeout(() => {
                        this.displayEvidence(result.evidence);
                    }, 800); // Wait for fade-in effect to complete + buffer
                }
            } else {
                console.error('❌ Analysis failed:', result.error);
                let errorMessage = result.error || 'Analysis failed';
                
                // Provide more specific error messages
                if (errorMessage.includes('No video uploaded')) {
                    errorMessage = 'No video found. Please upload a video first, then try analysis again.';
                } else if (errorMessage.includes('session')) {
                    errorMessage = 'Session error. Please refresh the page and try again.';
                }
                
                this.showError(errorMessage);
            }
        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.hideLoadingModal();
            
            let errorMessage = error.message;
            
            // Provide more specific error messages
            if (errorMessage.includes('JSON')) {
                errorMessage = 'Server response error. Please try again or contact support.';
            } else if (errorMessage.includes('fetch')) {
                errorMessage = 'Network error. Please check your connection and try again.';
            }
            
            this.showError('Analysis failed: ' + errorMessage);
        }
    }

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
        
        const uploadSection = document.getElementById('uploadSection');
        const chatInterface = document.getElementById('chatInterface');
        
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
        
        // Enable chat input
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
        
        console.log('✅ Chat interface should now be visible');
    }

    minimizeChat() {
        console.log('📱 Minimizing chat interface...');
        
        const uploadSection = document.getElementById('uploadSection');
        const chatInterface = document.getElementById('chatInterface');
        
        // Hide chat interface with animation
        chatInterface.style.transition = 'all 0.3s ease';
        chatInterface.style.opacity = '0';
        chatInterface.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            chatInterface.style.display = 'none';
            
            // Show upload section
            uploadSection.style.display = 'block';
            uploadSection.style.opacity = '0';
            uploadSection.style.transform = 'translateY(-20px)';
            
            setTimeout(() => {
                uploadSection.style.transition = 'all 0.3s ease';
                uploadSection.style.opacity = '1';
                uploadSection.style.transform = 'translateY(0)';
            }, 50);
        }, 300);
        
        console.log('✅ Upload interface should now be visible');
    }

    async sendMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput.value.trim();

        if (!message || this.isTyping) return;

        // Add user message
        this.addChatMessage('user', message);
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        // Disable input while processing
        chatInput.disabled = true;
        const sendBtn = document.getElementById('sendBtn');
        sendBtn.disabled = true;

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            const result = await response.json();

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
                this.addChatMessage('ai', 'Sorry, I encountered an error. Please try again.');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addChatMessage('ai', 'Sorry, I encountered an error. Please try again.');
        }
        
        // Re-enable input
        chatInput.disabled = false;
        sendBtn.disabled = false;
        chatInput.focus();
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

        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        // Show message with fade-in effect
        const messageText = messageDiv.querySelector('.message-text');
        this.showMessageWithEffect(messageText, formattedMessage);
    }

    showMessageWithEffect(element, text) {
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
        `
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
                this.resetUploadInternal();
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
            
            // Show video preview with demo video URL
            const videoPreview = document.getElementById('videoPreview');
            const previewVideo = document.getElementById('previewVideo');
            
            // Set the demo video source
            previewVideo.src = '/demo-video';
            
            // Show the preview section
            videoPreview.style.display = 'block';
            
            // Update preview header for demo video
            const previewHeader = videoPreview.querySelector('.preview-header h3');
            if (previewHeader) {
                previewHeader.textContent = 'Demo Video Preview';
            }
            
            const previewDescription = videoPreview.querySelector('.preview-header p');
            if (previewDescription) {
                previewDescription.textContent = 'Preview the BMW M4 demo video before using it for analysis';
            }
            
            console.log('🎬 Demo video preview shown');
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
                
                // Set the current file and mark as uploaded
                this.currentFile = demoFile;
                this.fileUploaded = true;
                
                this.showFileInfo(demoFile);
                
                // Update file info with actual data from server
                if (result.filename) {
                    const fileInfo = document.getElementById('fileInfo');
                    const fileName = fileInfo.querySelector('#fileName');
                    if (fileName) {
                        fileName.textContent = result.filename;
                    }
                }
                
                // Show success message
                this.showSuccess('Demo video loaded successfully! 🎬');
                
                // Show analysis form and update button state
                this.showAnalysisForm();
                this.updateAnalysisButtonState();
            } else {
                this.hideProgress();
                this.showError(result.error || 'Failed to load demo video');
            }
        } catch (error) {
            this.hideProgress();
            this.showError('Failed to load demo video: ' + error.message);
        }
    }

    initializeModelSelection() {
        // Initialize model selection with current model
        this.updateModelInfo();
        
        // Add event listener for model switching
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            // Remove any existing event listeners to prevent duplicates
            modelSelect.removeEventListener('change', this.handleModelChange);
            
            // Add the event listener
            this.handleModelChange = this.handleModelChange.bind(this);
            modelSelect.addEventListener('change', this.handleModelChange);
            
            // Add click event to prevent bubbling
            modelSelect.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }
        
        // Also prevent clicks on the model selection container from bubbling
        const modelSelection = document.querySelector('.model-selection');
        if (modelSelection) {
            modelSelection.addEventListener('click', (e) => {
                e.stopPropagation();
            });
        }
    }
    
    handleModelChange(e) {
        // Prevent event bubbling
        e.stopPropagation();
        this.switchModel(e.target.value);
    }
    
    async switchModel(modelName) {
        try {
            console.log(`🔄 Switching to model: ${modelName}`);
            
            // Show loading state
            this.showModelLoadingState(modelName);
            
            // Call API to switch model
            const response = await fetch('/api/switch-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ model: modelName })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`✅ Successfully switched to ${modelName}`);
                this.updateModelInfo();
                this.showSuccess(`Successfully switched to ${result.model_name}`);
            } else {
                console.error(`❌ Failed to switch to ${modelName}: ${result.error}`);
                this.showError(`Failed to switch model: ${result.error}`);
                // Revert selection
                this.revertModelSelection();
            }
            
        } catch (error) {
            console.error('❌ Model switch error:', error);
            this.showError('Failed to switch model. Please try again.');
            this.revertModelSelection();
        }
    }
    
    showModelLoadingState(modelName) {
        const modelInfo = document.getElementById('modelInfo');
        if (modelInfo) {
            modelInfo.innerHTML = `<small>🔄 Switching to ${modelName}...</small>`;
        }
        
        // Disable select during switch
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.disabled = true;
        }
    }
    
    updateModelInfo() {
        const modelSelect = document.getElementById('modelSelect');
        const modelInfo = document.getElementById('modelInfo');
        
        if (modelSelect && modelInfo) {
            const selectedModel = modelSelect.value;
            const modelDescriptions = {
                'minicpm': 'Fast and efficient vision-language model for quick analysis',
                'qwen25vl': 'Advanced multimodal model with enhanced video understanding capabilities',
                'qwen25vl_32b': 'High-performance 32B parameter model with superior video analysis capabilities'
            };
            
            // Update the model info text
            modelInfo.innerHTML = `<small>${modelDescriptions[selectedModel] || 'Select an AI model for video analysis'}</small>`;
            
            // Add visual feedback for selected model
            modelSelect.classList.remove('model-selected-minicpm', 'model-selected-qwen25vl', 'model-selected-qwen25vl_32b');
            modelSelect.classList.add(`model-selected-${selectedModel}`);
            
            // Update the model selection container styling
            const modelSelection = document.querySelector('.model-selection');
            if (modelSelection) {
                modelSelection.classList.remove('model-selected-minicpm', 'model-selected-qwen25vl', 'model-selected-qwen25vl_32b');
                modelSelection.classList.add(`model-selected-${selectedModel}`);
            }
        }
    }
    
    revertModelSelection() {
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            // Revert to previous selection (you might want to store the previous value)
            modelSelect.value = 'minicpm';
            this.updateModelInfo();
        }
        
        // Re-enable select
        modelSelect.disabled = false;
    }

    // Add missing functions that are called from HTML
    minimizeChat() {
        console.log('📱 Minimizing chat interface...');
        const chatInterface = document.getElementById('chatInterface');
        const uploadSection = document.getElementById('uploadSection');
        
        if (chatInterface && uploadSection) {
            // Hide chat interface
            chatInterface.style.transition = 'all 0.3s ease';
            chatInterface.style.opacity = '0';
            chatInterface.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                chatInterface.style.display = 'none';
                
                // Show upload section
                uploadSection.style.display = 'block';
                uploadSection.style.opacity = '0';
                uploadSection.style.transform = 'translateY(-20px)';
                
                setTimeout(() => {
                    uploadSection.style.transition = 'all 0.3s ease';
                    uploadSection.style.opacity = '1';
                    uploadSection.style.transform = 'translateY(0)';
                }, 50);
            }, 300);
        }
    }

    testUpload() {
        console.log('🧪 Testing upload functionality...');
        // Create a test file object
        const testFile = new File(['test'], 'test.mp4', { type: 'video/mp4' });
        this.currentFile = testFile;
        this.showFileInfo(testFile);
        this.showSuccess('Test file loaded successfully! You can now test the analysis.');
    }

    resetUpload() {
        console.log('🔄 Resetting upload...');
        // Call the actual reset method
        this.resetUploadInternal();
    }

    resetUploadInternal() {
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
        
        // Hide analysis form
        this.hideAnalysisForm();
        
        // Clear current file reference
        this.currentFile = null;
        this.fileUploaded = false;
        
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
        this.cleanupOldUploadsInternal();
        
        console.log('🔄 Upload interface reset');
    }

    cleanupOldUploads() {
        console.log('🧹 Cleaning up old uploads...');
        // Call the actual cleanup method
        this.cleanupOldUploadsInternal();
    }

    async cleanupOldUploadsInternal() {
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

    async uploadAndAnalyze() {
        console.log('🚀 Starting upload and analysis process...');
        
        if (!this.currentFile) {
            this.showError('Please select a video file first');
            return;
        }

        try {
            // Get selected analysis mode and user focus
            const selectedMode = window.selectedAnalysisMode || 'standard';
            const userFocus = document.getElementById('userFocus')?.value || 'Analyze this video comprehensively';
            
            console.log('🎯 Analysis mode:', selectedMode);
            console.log('🎯 User focus:', userFocus);
            
            // First upload the file
            console.log('📤 Uploading file...');
            await this.uploadFile(this.currentFile);
            
            // Then start analysis based on selected mode
            console.log('🔍 Starting analysis with mode:', selectedMode);
            
            if (selectedMode === 'ultra_accurate') {
                await this.startUltraAccurateAnalysis(userFocus);
            } else {
                await this.startAnalysis();
            }
            
        } catch (error) {
            console.error('❌ Upload and analysis failed:', error);
            this.showError('Process failed: ' + error.message);
        }
    }

    async startUltraAccurateAnalysis(userFocus) {
        console.log('🚀 Starting ultra-accurate analysis...');
        
        try {
            // Show progress section
            this.showProgressSection();
            this.updateProgress(10, 'Initializing ultra-accurate AI service...');
            
            // Make API call to ultra-accurate endpoint
            const response = await fetch('/analyze-ultra-accurate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_focus: userFocus
                })
            });
            
            this.updateProgress(30, 'Processing video with ultra-accurate analysis...');
            
            const result = await response.json();
            
            if (result.success) {
                this.updateProgress(100, 'Ultra-accurate analysis completed successfully!');
                
                // Hide progress section
                this.hideProgressSection();
                
                // Show results
                this.showResults(result);
                
                // Show ultra-accurate Q&A section if available
                const ultraAccurateQASection = document.getElementById('ultraAccurateQASection');
                if (ultraAccurateQASection) {
                    ultraAccurateQASection.style.display = 'block';
                }
                
                console.log('✅ Ultra-accurate analysis completed successfully!');
                
            } else {
                throw new Error(result.error || 'Ultra-accurate analysis failed');
            }
            
        } catch (error) {
            console.error('❌ Ultra-accurate analysis error:', error);
            this.hideProgressSection();
            this.showError('Ultra-accurate analysis failed: ' + error.message);
        }
    }

    showProgressSection() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'block';
        }
    }

    hideProgressSection() {
        const progressSection = document.getElementById('progressSection');
        if (progressSection) {
            progressSection.style.display = 'none';
        }
    }

    updateProgress(percent, status) {
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const progressStatus = document.getElementById('progressStatus');
        
        if (progressBar) {
            progressBar.style.width = percent + '%';
        }
        
        if (progressText) {
            progressText.textContent = percent + '%';
        }
        
        if (progressStatus) {
            progressStatus.textContent = status;
        }
    }

    showResults(result) {
        // Show results section
        const resultsSection = document.getElementById('resultsSection');
        if (resultsSection) {
            resultsSection.style.display = 'block';
        }
        
        // Display analysis text
        const analysisText = document.getElementById('analysisText');
        if (analysisText && result.analysis) {
            analysisText.innerHTML = result.analysis.replace(/\n/g, '<br>');
        }
        
        // Display video duration
        const durationInfo = document.getElementById('durationInfo');
        if (durationInfo && result.video_duration) {
            const minutes = Math.floor(result.video_duration / 60);
            const seconds = Math.floor(result.video_duration % 60);
            durationInfo.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
        }
        
        // Display timestamps
        const timestampsList = document.getElementById('timestampsList');
        if (timestampsList && result.timestamps) {
            timestampsList.innerHTML = '';
            if (result.timestamps.length > 0) {
                result.timestamps.forEach(timestamp => {
                    const li = document.createElement('li');
                    li.className = 'list-group-item';
                    li.innerHTML = `<i class="fas fa-clock text-primary"></i> ${timestamp}`;
                    timestampsList.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.className = 'list-group-item text-muted';
                li.textContent = 'No timestamps found';
                timestampsList.appendChild(li);
            }
        }
        
        // Show ultra-accurate metadata if available
        if (result.ultra_accurate_mode) {
            const resultsSection = document.getElementById('resultsSection');
            if (resultsSection) {
                const ultraAccurateInfo = document.createElement('div');
                ultraAccurateInfo.className = 'alert alert-warning mt-3';
                ultraAccurateInfo.innerHTML = `
                    <i class="fas fa-rocket text-warning"></i>
                    <strong>Ultra-Accurate Analysis Completed!</strong><br>
                    <small>
                        • GPU Optimization: ${result.gpu_optimization}<br>
                        • Analysis Quality: ${result.analysis_quality}<br>
                        • Multi-scale Analysis: ${result.multi_scale_analysis ? '✅' : '❌'}<br>
                        • Cross-validation: ${result.cross_validation ? '✅' : '❌'}<br>
                        • Max Video Duration: ${result.max_video_duration}
                    </small>
                `;
                resultsSection.querySelector('.card-body').appendChild(ultraAccurateInfo);
            }
        }
    }
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

// Test if functions are available
console.log('🔍 Testing global function availability...');
console.log('window.startAnalysis:', typeof window.startAnalysis);
console.log('window.uploadDemoVideo:', typeof window.uploadDemoVideo);
console.log('window.testUpload:', typeof window.testUpload);
console.log('window.resetUpload:', typeof window.resetUpload);

window.closeEvidenceModal = () => window.videoDetective.closeEvidenceModal();
window.cleanupSession = () => window.videoDetective.cleanupSession();
window.uploadSelectedVideo = () => window.videoDetective.uploadSelectedVideo();
window.uploadDemoVideo = () => window.videoDetective.uploadDemoVideo();
window.startAnalysis = () => window.videoDetective.startAnalysis();
window.minimizeChat = () => window.videoDetective.minimizeChat();
window.testUpload = () => window.videoDetective.testUpload();
window.resetUpload = () => window.videoDetective.resetUpload();
window.cleanupOldUploads = () => window.videoDetective.cleanupOldUploads();
window.uploadAndAnalyze = () => window.videoDetective.uploadAndAnalyze();

// Test again after setting them
console.log('🔍 After setting global functions...');
console.log('window.startAnalysis:', typeof window.startAnalysis);
console.log('window.uploadDemoVideo:', typeof window.uploadDemoVideo);
console.log('window.testUpload:', typeof window.testUpload);
console.log('window.resetUpload:', typeof window.resetUpload);

console.log('🚀 AI Video Detective Pro is ready!');