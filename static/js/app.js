// AI Video Detective - Modern Professional JavaScript Application

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
        this.setupNavigation();
        this.checkSessionStatus();
        this.setupPageCleanup();
        this.initializeModelSelection();
        console.log('✅ AI Video Detective Pro initialized successfully!');
    }

    setupEventListeners() {
        // File upload
        const videoFile = document.getElementById('videoFile');
        const uploadArea = document.getElementById('uploadArea');

        if (videoFile && uploadArea) {
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
                // Don't open file dialog if clicking on model selection or other interactive elements
                if (e.target.closest('.model-selection') || 
                    e.target.closest('.upload-buttons') || 
                    e.target.closest('.upload-features')) {
                    return;
                }
                videoFile.click();
            });
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
        }
    }

    setupNavigation() {
        const navButtons = document.querySelectorAll('.nav-btn');
        const sections = document.querySelectorAll('.section');

        navButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetSection = button.getAttribute('data-section');
                
                // Update active nav button
                navButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Show target section
                sections.forEach(section => {
                    section.classList.remove('active');
                    if (section.id === targetSection) {
                        section.classList.add('active');
                    }
                });
            });
        });
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
            const response = await fetch('/cleanup-uploads', {
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
    }

    isValidVideoFile(file) {
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/webm', 'video/x-matroska'];
        return validTypes.includes(file.type);
    }

    showFileInfo(file) {
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        
        if (fileInfo && fileName && fileSize) {
            fileName.textContent = file.name;
            fileSize.textContent = this.formatFileSize(file.size);
            fileInfo.style.display = 'block';
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
        
        if (videoPreview && previewVideo) {
            // Create object URL for video preview
            const videoUrl = URL.createObjectURL(file);
            previewVideo.src = videoUrl;
            
            // Show the preview section
            videoPreview.style.display = 'block';
            
            // Scroll to preview
            videoPreview.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            console.log('🎬 Video preview shown for:', file.name);
        }
    }

    async uploadSelectedVideo() {
        // Hide the preview
        const videoPreview = document.getElementById('videoPreview');
        if (videoPreview) {
            videoPreview.style.display = 'none';
        }
        
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
                this.analyzeVideo();
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
        const progressFill = progress ? progress.querySelector('.progress-fill') : null;
        
        if (progress && progressFill) {
            progress.style.display = 'block';
            progressFill.style.width = '0%';
            
            // Simulate progress
            let width = 0;
            const interval = setInterval(() => {
                if (width >= 90) {
                    clearInterval(interval);
                } else {
                    width += Math.random() * 10;
                    progressFill.style.width = width + '%';
                }
            }, 200);
        }
    }

    hideProgress() {
        const progress = document.getElementById('uploadProgress');
        const progressFill = progress ? progress.querySelector('.progress-fill') : null;
        
        if (progressFill) {
            progressFill.style.width = '100%';
            setTimeout(() => {
                if (progress) {
                    progress.style.display = 'none';
                }
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

            const result = await response.json();
            console.log('📊 Analysis response:', result);
            
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
                this.showError(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('❌ Analysis error:', error);
            this.hideLoadingModal();
            this.showError('Analysis failed: ' + error.message);
        }
    }

    displayEvidence(evidence, title = 'Visual Evidence') {
        const chatMessages = document.getElementById('chatMessages');
        
        if (!chatMessages) return;
        
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
        
        if (modal && modalImage && modalInfo) {
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
    }

    closeEvidenceModal() {
        const modal = document.getElementById('evidenceModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    formatTimestamp(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    showChatInterface() {
        console.log('💬 Showing chat interface...');
        
        const uploadSection = document.getElementById('upload');
        const chatInterface = document.getElementById('chat');
        
        // Hide upload section with animation
        if (uploadSection) {
            uploadSection.style.transition = 'all 0.3s ease';
            uploadSection.style.opacity = '0';
            uploadSection.style.transform = 'translateY(-20px)';
            
            setTimeout(() => {
                uploadSection.style.display = 'none';
                
                // Show chat interface
                if (chatInterface) {
                    chatInterface.style.display = 'block';
                    chatInterface.style.opacity = '0';
                    chatInterface.style.transform = 'translateY(20px)';
                    
                    setTimeout(() => {
                        chatInterface.style.transition = 'all 0.3s ease';
                        chatInterface.style.opacity = '1';
                        chatInterface.style.transform = 'translateY(0)';
                    }, 50);
                }
            }, 300);
        }
        
        // Enable chat input
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        
        if (chatInput) {
            chatInput.disabled = false;
        }
        if (sendBtn) {
            sendBtn.disabled = false;
        }
        if (chatInput) {
            chatInput.focus();
        }
        
        console.log('✅ Chat interface should now be visible');
    }

    async sendMessage() {
        const chatInput = document.getElementById('chatInput');
        const message = chatInput ? chatInput.value.trim() : '';

        if (!message || this.isTyping) return;

        // Add user message
        this.addChatMessage('user', message);
        if (chatInput) {
            chatInput.value = '';
            chatInput.style.height = 'auto';
        }
        
        // Disable input while processing
        const sendBtn = document.getElementById('sendBtn');
        if (chatInput && sendBtn) {
            chatInput.disabled = true;
            sendBtn.disabled = true;
        }

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
        if (chatInput) {
            chatInput.disabled = false;
        }
        if (sendBtn) {
            sendBtn.disabled = false;
        }
        if (chatInput) {
            chatInput.focus();
        }
    }

    showTypingIndicator() {
        this.isTyping = true;
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.display = 'block';
        }
        
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.style.display = 'none';
        }
    }

    addChatMessage(type, message) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

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
        if (!chatMessages) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

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
        // Start with opacity 0 and add fade-in effect
        if (element) {
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
        
        if (modal && loadingMessage) {
            loadingMessage.textContent = message;
            modal.style.display = 'flex';
            
            // Animate loading steps
            this.animateLoadingSteps();
        }
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
        if (modal) {
            modal.style.display = 'none';
        }
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
            const response = await fetch('/session/cleanup', {
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
            const response = await fetch('/session/status');
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

    resetUpload() {
        // Clear current file
        const videoFile = document.getElementById('videoFile');
        if (videoFile) {
            videoFile.value = '';
        }
        
        // Hide file info
        const fileInfo = document.getElementById('fileInfo');
        if (fileInfo) {
            fileInfo.style.display = 'none';
        }
        
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
        const uploadSection = document.getElementById('upload');
        const chatInterface = document.getElementById('chat');
        
        if (chatInterface) {
            chatInterface.style.transition = 'all 0.3s ease';
            chatInterface.style.opacity = '0';
            chatInterface.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                chatInterface.style.display = 'none';
                
                if (uploadSection) {
                    uploadSection.style.display = 'block';
                    uploadSection.style.opacity = '0';
                    uploadSection.style.transform = 'translateY(-20px)';
                    
                    setTimeout(() => {
                        uploadSection.style.transition = 'all 0.3s ease';
                        uploadSection.style.opacity = '1';
                        uploadSection.style.transform = 'translateY(0)';
                    }, 50);
                }
            }, 300);
        }
        
        // Clear chat messages
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
        
        // Disable chat input
        const chatInput = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendBtn');
        if (chatInput) {
            chatInput.disabled = true;
        }
        if (sendBtn) {
            sendBtn.disabled = true;
        }
        if (chatInput) {
            chatInput.style.height = 'auto';
        }
        
        // Reset analysis state
        this.analysisComplete = false;
        
        // Clean up old uploads when returning to home screen
        this.cleanupOldUploads();
        
        console.log('🔄 Upload interface reset');
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
        if (modelSelect) {
            modelSelect.disabled = false;
        }
    }

    // Streaming functionality
    toggleStreamingInterface() {
        const interface = document.getElementById('streamingInterface');
        const toggleBtn = document.querySelector('.toggle-btn');
        
        if (interface && toggleBtn) {
            if (interface.classList.contains('hidden')) {
                interface.classList.remove('hidden');
                toggleBtn.innerHTML = '<i class="fas fa-lock"></i> Hide Streaming Interface';
                this.connectWebSocket();
            } else {
                interface.classList.add('hidden');
                toggleBtn.innerHTML = '<i class="fas fa-rocket"></i> Enable Streaming Interface';
                this.disconnectWebSocket();
            }
        }
    }
    
    connectWebSocket() {
        try {
            // For now, we'll use polling instead of WebSocket
            // In a real implementation, you'd connect to a WebSocket server
            this.updateConnectionStatus('connected');
            this.startStatusPolling();
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.updateConnectionStatus('disconnected');
        }
    }
    
    disconnectWebSocket() {
        this.updateConnectionStatus('disconnected');
        this.stopStatusPolling();
    }
    
    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            if (status === 'connected') {
                statusElement.className = 'connection-status connected';
                statusElement.innerHTML = '<i class="fas fa-circle"></i> WebSocket Connected';
            } else {
                statusElement.className = 'connection-status disconnected';
                statusElement.innerHTML = '<i class="fas fa-circle"></i> WebSocket Disconnected';
            }
        }
    }
    
    startStatusPolling() {
        if (this.statusPollingInterval) return;
        
        this.statusPollingInterval = setInterval(() => {
            if (this.currentStreamId && this.isStreaming) {
                this.updateStreamStatus();
                this.updateStreamEvents();
            }
        }, 2000); // Poll every 2 seconds
    }
    
    stopStatusPolling() {
        if (this.statusPollingInterval) {
            clearInterval(this.statusPollingInterval);
            this.statusPollingInterval = null;
        }
    }
    
    async startStream() {
        const streamId = document.getElementById('streamId')?.value;
        const videoSource = document.getElementById('videoSource')?.value;
        
        if (!streamId || !videoSource) {
            alert('Please enter both Stream ID and Video Source');
            return;
        }
        
        try {
            const response = await fetch('/api/stream/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    stream_id: streamId,
                    video_source: videoSource,
                    config: {
                        fps_target: 30,
                        event_detection: true,
                        ai_analysis_enabled: true,
                        analysis_interval: 1
                    }
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.currentStreamId = streamId;
                this.isStreaming = true;
                
                // Update UI
                const startBtn = document.querySelector('.stream-controls button:first-of-type');
                if (startBtn) {
                    startBtn.textContent = '⏸️ Pause Stream';
                    startBtn.onclick = () => this.pauseStream();
                }
                
                // Start video display if it's a camera
                if (videoSource === '0' || videoSource === '1') {
                    this.startCameraDisplay(parseInt(videoSource));
                }
                
                alert(`Stream started successfully: ${streamId}`);
                
                // Start status updates
                this.updateStreamStatus();
                this.updateStreamEvents();
                
            } else {
                alert(`Failed to start stream: ${result.error}`);
            }
            
        } catch (error) {
            console.error('Stream start failed:', error);
            alert('Failed to start stream. Check console for details.');
        }
    }
    
    async stopStream() {
        if (!this.currentStreamId) {
            alert('No active stream to stop');
            return;
        }
        
        try {
            const response = await fetch(`/api/stream/stop/${this.currentStreamId}`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                // Reset UI
                const startBtn = document.querySelector('.stream-controls button:first-of-type');
                if (startBtn) {
                    startBtn.textContent = 'Start Stream';
                    startBtn.onclick = () => this.startStream();
                }
                
                // Stop camera display
                this.stopCameraDisplay();
                
                // Reset variables
                this.currentStreamId = null;
                this.isStreaming = false;
                
                // Reset metrics
                this.resetStreamMetrics();
                
                alert('Stream stopped successfully');
                
            } else {
                alert(`Failed to stop stream: ${result.error}`);
            }
            
        } catch (error) {
            console.error('Stream stop failed:', error);
            alert('Failed to stop stream. Check console for details.');
        }
    }
    
    pauseStream() {
        // For now, just stop the stream
        this.stopStream();
    }
    
    async updateStreamStatus() {
        if (!this.currentStreamId) return;
        
        try {
            const response = await fetch(`/api/stream/status/${this.currentStreamId}`);
            const result = await response.json();
            
            if (result.success && result.status) {
                const status = result.status;
                const metrics = status.metrics || {};
                
                // Update UI metrics
                const currentFps = document.getElementById('currentFps');
                const targetFps = document.getElementById('targetFps');
                const currentLatency = document.getElementById('currentLatency');
                const eventsCount = document.getElementById('eventsCount');
                const aiAnalysisCount = document.getElementById('aiAnalysisCount');
                
                if (currentFps) currentFps.textContent = metrics.fps_current?.toFixed(1) || '0.0';
                if (targetFps) targetFps.textContent = metrics.fps_target || '30';
                if (currentLatency) currentLatency.textContent = `${metrics.latency_ms?.toFixed(1) || '0'}ms`;
                if (eventsCount) eventsCount.textContent = status.event_count || '0';
                if (aiAnalysisCount) aiAnalysisCount.textContent = status.ai_analysis_count || '0';
            }
            
        } catch (error) {
            console.error('Failed to update stream status:', error);
        }
    }
    
    async updateStreamEvents() {
        if (!this.currentStreamId) return;
        
        try {
            const response = await fetch(`/api/stream/events/${this.currentStreamId}`);
            const result = await response.json();
            
            if (result.success && result.events) {
                this.displayStreamEvents(result.events);
            }
            
        } catch (error) {
            console.error('Failed to update stream events:', error);
        }
    }
    
    displayStreamEvents(events) {
        const eventsList = document.getElementById('streamEventsList');
        if (!eventsList) return;
        
        if (!events || events.length === 0) {
            eventsList.innerHTML = '<p style="text-align: center; opacity: 0.7;">No events detected yet.</p>';
            return;
        }
        
        // Sort events by timestamp (newest first)
        const sortedEvents = events.sort((a, b) => b.timestamp - a.timestamp);
        
        // Display last 10 events
        const recentEvents = sortedEvents.slice(0, 10);
        
        eventsList.innerHTML = recentEvents.map(event => {
            const eventTime = new Date(event.timestamp * 1000).toLocaleTimeString();
            const isAIAnalysis = event.event_type === 'ai_analysis';
            
            return `
                <div class="event-item ${isAIAnalysis ? 'ai-analysis' : ''}">
                    <div class="event-header">
                        <span class="event-type">${event.event_type.toUpperCase()}</span>
                        <span class="event-time">${eventTime}</span>
                    </div>
                    <div class="event-content">
                        Confidence: ${(event.confidence * 100).toFixed(1)}%
                        ${event.metadata ? `<br>Details: ${JSON.stringify(event.metadata)}` : ''}
                        ${event.ai_analysis ? `<div class="ai-analysis-content">🧠 AI Analysis: ${event.ai_analysis}</div>` : ''}
                    </div>
                </div>
            `;
        }).join('');
    }
    
    resetStreamMetrics() {
        const currentFps = document.getElementById('currentFps');
        const targetFps = document.getElementById('targetFps');
        const currentLatency = document.getElementById('currentLatency');
        const eventsCount = document.getElementById('eventsCount');
        const aiAnalysisCount = document.getElementById('aiAnalysisCount');
        const streamEventsList = document.getElementById('streamEventsList');
        
        if (currentFps) currentFps.textContent = '0.0';
        if (targetFps) targetFps.textContent = '30';
        if (currentLatency) currentLatency.textContent = '0ms';
        if (eventsCount) eventsCount.textContent = '0';
        if (aiAnalysisCount) aiAnalysisCount.textContent = '0';
        if (streamEventsList) {
            streamEventsList.innerHTML = '<p style="text-align: center; opacity: 0.7;">No events detected yet. Start a stream to see live analysis.</p>';
        }
    }
    
    startCameraDisplay(cameraIndex) {
        try {
            const video = document.getElementById('streamVideo');
            if (!video) return;
            
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        deviceId: cameraIndex ? { exact: cameraIndex } : undefined 
                    } 
                })
                .then(stream => {
                    this.cameraStream = stream;
                    video.srcObject = stream;
                    video.play();
                })
                .catch(error => {
                    console.error('Camera access failed:', error);
                    // Fallback to file input for video files
                    const fileInput = document.createElement('input');
                    fileInput.type = 'file';
                    fileInput.accept = 'video/*';
                    fileInput.onchange = (e) => {
                        const file = e.target.files[0];
                        if (file) {
                            const url = URL.createObjectURL(file);
                            video.src = url;
                            video.play();
                        }
                    };
                    fileInput.click();
                });
            }
        } catch (error) {
            console.error('Camera display failed:', error);
        }
    }
    
    stopCameraDisplay() {
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
            this.cameraStream = null;
        }
        
        const video = document.getElementById('streamVideo');
        if (video) {
            video.src = '';
            video.srcObject = null;
        }
    }

    // Session management
    async refreshSessions() {
        try {
            const sessionsList = document.getElementById('sessionsList');
            if (sessionsList) {
                sessionsList.innerHTML = '<p>Refreshing sessions...</p>';
                
                const response = await fetch('/api/sessions');
                const result = await response.json();
                
                if (result.success) {
                    this.displaySessions(result.sessions);
                } else {
                    sessionsList.innerHTML = '<p>Failed to load sessions</p>';
                }
            }
        } catch (error) {
            console.error('Failed to refresh sessions:', error);
            const sessionsList = document.getElementById('sessionsList');
            if (sessionsList) {
                sessionsList.innerHTML = '<p>Error loading sessions</p>';
            }
        }
    }
    
    async cleanupSessions() {
        try {
            const response = await fetch('/api/sessions/cleanup', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess('Sessions cleaned up successfully');
                this.refreshSessions();
            } else {
                this.showError('Failed to cleanup sessions: ' + result.error);
            }
        } catch (error) {
            console.error('Failed to cleanup sessions:', error);
            this.showError('Error cleaning up sessions');
        }
    }
    
    displaySessions(sessions) {
        const sessionsList = document.getElementById('sessionsList');
        if (!sessionsList) return;
        
        if (!sessions || sessions.length === 0) {
            sessionsList.innerHTML = '<p>No sessions found</p>';
            return;
        }
        
        const sessionsHtml = sessions.map(session => `
            <div class="session-item">
                <h4>${session.name || 'Unnamed Session'}</h4>
                <p>Created: ${new Date(session.created_at).toLocaleString()}</p>
                <p>Status: ${session.status}</p>
            </div>
        `).join('');
        
        sessionsList.innerHTML = sessionsHtml;
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
window.closeEvidenceModal = () => window.videoDetective.closeEvidenceModal();
window.cleanupSession = () => window.videoDetective.cleanupSession();
window.uploadSelectedVideo = () => window.videoDetective.uploadSelectedVideo();
window.toggleStreamingInterface = () => window.videoDetective.toggleStreamingInterface();
window.startStream = () => window.videoDetective.startStream();
window.stopStream = () => window.videoDetective.stopStream();
window.refreshSessions = () => window.videoDetective.refreshSessions();
window.cleanupSessions = () => window.videoDetective.cleanupSessions();

console.log('🚀 AI Video Detective Pro is ready!');