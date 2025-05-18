// Modal Features Implementation
document.addEventListener('DOMContentLoaded', function() {
    // Get references to modal elements with more reliable selectors
    const modalCameraFeed = document.getElementById('modalCameraFeed');
    const playButton = document.querySelector('#cameraModal button:has(i.ri-play-fill), #cameraModal button:has(i.ri-pause-fill)');
    const rewindButton = document.querySelector('#cameraModal button:has(i.ri-rewind-mini-fill)');
    const speedButton = document.querySelector('#cameraModal button:has(i.ri-speed-mini-fill)');
    const recordButton = document.querySelector('#cameraModal button:has(i.ri-record-circle-line), #cameraModal button:has(i.ri-stop-circle-line)');
    const timelineSlider = document.querySelector('#cameraModal .relative.w-full.h-1');
    const timelineHandle = document.querySelector('#cameraModal .absolute.left-\\[75\\%\\]');
    const timeDisplay = document.querySelector('#cameraModal .text-sm:first-of-type');
    
    // Find quality button using text content instead of :contains selector
    const qualityButton = findElementWithText('#cameraModal button', 'High') || 
                         document.querySelector('#cameraModal .flex.items-center.justify-between button');
    
    // Helper function to find element containing specific text
    function findElementWithText(selector, text) {
        const elements = document.querySelectorAll(selector);
        for (const element of elements) {
            if (element.textContent.includes(text)) {
                return element;
            }
        }
        return null;
    }
    
    // Log element discovery for debugging
    console.log("Modal elements found:", {
        modalCameraFeed: !!modalCameraFeed,
        playButton: !!playButton,
        rewindButton: !!rewindButton,
        speedButton: !!speedButton,
        recordButton: !!recordButton,
        timelineSlider: !!timelineSlider,
        timelineHandle: !!timelineHandle,
        timeDisplay: !!timeDisplay,
        qualityButton: !!qualityButton
    });
    
    // State variables
    let isPlaying = true;
    let playbackSpeed = 1.0;
    let isRecording = false;
    let recordingStartTime = null;
    let recordingInterval = null;
    let timelineUpdateInterval = null;
    let currentPosition = 75; // Percentage position in timeline
    let isDraggingTimeline = false;
    
    // Settings states
    const settingsStates = {
        motionDetection: true,
        faceRecognition: true,
        audioRecording: false,
        aiDetection: false
    };
    
    // Quality options
    const qualityOptions = [
        { label: 'Low (480p)', value: '480p' },
        { label: 'Medium (720p)', value: '720p' },
        { label: 'High (1080p)', value: '1080p' },
        { label: 'Ultra (2160p)', value: '4k' }
    ];
    let currentQuality = 'High (1080p)';
    
    // Safely get element by fallback selectors - tries multiple selectors in sequence
    function safeQuerySelector(selectors) {
        if (typeof selectors === 'string') {
            return document.querySelector(selectors);
        }
        
        // Try each selector in turn
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) return element;
        }
        
        return null;
    }
    
    // Video Controls Implementation
    function initVideoControls() {
        // Find play button if not already found
        if (!playButton) {
            const alternativeSelectors = [
                '#cameraModal .ri-play-fill',
                '#cameraModal button i.ri-play-fill',
                '#cameraModal button i.ri-pause-fill'
            ].map(s => `${s}`);
            
            const foundPlayButton = alternativeSelectors.map(s => document.querySelector(s)?.closest('button')).find(Boolean);
            
            if (foundPlayButton) {
                foundPlayButton.addEventListener('click', togglePlayback);
                console.log("Play button found and initialized with alternative selector");
            }
        } else {
            // Play/Pause button
            playButton.addEventListener('click', togglePlayback);
        }
        
        // Rewind button (10 seconds back)
        if (rewindButton) {
            rewindButton.addEventListener('click', function() {
                rewindVideo(10);
            });
        } else {
            const foundRewindButton = document.querySelector('#cameraModal .ri-rewind-mini-fill')?.closest('button');
            if (foundRewindButton) {
                foundRewindButton.addEventListener('click', function() {
                    rewindVideo(10);
                });
                console.log("Rewind button found with alternative selector");
            }
        }
        
        // Speed button (cycle through speeds)
        if (speedButton) {
            speedButton.addEventListener('click', cyclePlaybackSpeed);
        } else {
            const foundSpeedButton = document.querySelector('#cameraModal .ri-speed-mini-fill')?.closest('button');
            if (foundSpeedButton) {
                foundSpeedButton.addEventListener('click', cyclePlaybackSpeed);
                console.log("Speed button found with alternative selector");
            }
        }
        
        // Recording button
        if (recordButton) {
            recordButton.addEventListener('click', toggleRecording);
        } else {
            const foundRecordButton = document.querySelector('#cameraModal .ri-record-circle-line')?.closest('button');
            if (foundRecordButton) {
                foundRecordButton.addEventListener('click', toggleRecording);
                console.log("Record button found with alternative selector");
            }
        }
        
        // Initialize timeline dragging if elements exist
        if (timelineSlider || document.querySelector('#cameraModal .relative.w-full.h-1')) {
            initTimelineDragging();
        }
    }
    
    // Play/Pause toggle
    function togglePlayback() {
        isPlaying = !isPlaying;
        
        // Find icon element safely
        const buttonElement = this || playButton || document.querySelector('#cameraModal button:has(.ri-play-fill), #cameraModal button:has(.ri-pause-fill)');
        if (!buttonElement) return;
        
        const icon = buttonElement.querySelector('i');
        if (!icon) return;
        
        if (!isPlaying) {
            // Switch to pause icon
            if (icon.classList.contains('ri-play-fill')) {
                icon.classList.remove('ri-play-fill');
                icon.classList.add('ri-pause-fill');
            }
            // Simulated pause - in real implementation would pause the stream
            showFeedbackToast('Video paused');
        } else {
            // Switch to play icon
            if (icon.classList.contains('ri-pause-fill')) {
                icon.classList.remove('ri-pause-fill');
                icon.classList.add('ri-play-fill');
            }
            // Simulated play - in real implementation would resume the stream
            showFeedbackToast('Video playing');
        }
    }
    
    // Rewind video by specified seconds
    function rewindVideo(seconds) {
        // In a real implementation, this would seek the video back
        currentPosition = Math.max(0, currentPosition - (seconds * 2)); // 2% per second as example
        updateTimelinePosition();
        updateTimeDisplay();
        showFeedbackToast(`Rewound ${seconds} seconds`);
    }
    
    // Cycle through playback speeds
    function cyclePlaybackSpeed() {
        const speeds = [0.5, 1.0, 1.5, 2.0];
        const currentIndex = speeds.indexOf(playbackSpeed);
        const nextIndex = (currentIndex + 1) % speeds.length;
        playbackSpeed = speeds[nextIndex];
        
        // Update the display
        showFeedbackToast(`Playback speed: ${playbackSpeed}x`);
    }
    
    // Toggle recording state
    function toggleRecording() {
        isRecording = !isRecording;
        
        // Find icon element safely
        const buttonElement = this || recordButton || document.querySelector('#cameraModal button:has(.ri-record-circle-line), #cameraModal button:has(.ri-stop-circle-line)');
        if (!buttonElement) return;
        
        const icon = buttonElement.querySelector('i');
        if (!icon) return;
        
        if (isRecording) {
            // Start recording
            icon.className = 'ri-stop-circle-line';
            buttonElement.classList.add('text-red-500', 'recording-indicator');
            
            // Record start time
            recordingStartTime = new Date();
            
            // Start recording duration display
            recordingInterval = setInterval(updateRecordingDuration, 1000);
            
            showFeedbackToast('Recording started');
            
            // Add pulsing effect to the modal title to indicate recording
            const modalTitle = document.getElementById('modalCameraTitle');
            if (modalTitle) {
                modalTitle.classList.add('recording-indicator');
            }
            
            // In a real implementation, this would start recording the stream
            const cameraId = window.currentModalCameraId ||
                           document.querySelector('#modalCameraId')?.textContent?.replace(/\D/g, '') ||
                           '1';
            
            // Optional: Make API call to start server-side recording
            /*
            fetch(`/api/start_recording/${cameraId}`, {
                method: 'POST'
            });
            */
            
        } else {
            // Stop recording
            icon.className = 'ri-record-circle-line';
            buttonElement.classList.remove('text-red-500', 'recording-indicator');
            
            // Stop interval
            clearInterval(recordingInterval);
            
            // Remove recording indicator from title
            const modalTitle = document.getElementById('modalCameraTitle');
            if (modalTitle) {
                modalTitle.classList.remove('recording-indicator');
            }
            
            // Reset time display
            const timeDisplayElement = timeDisplay || document.querySelector('#cameraModal .text-sm:first-of-type');
            if (timeDisplayElement) {
                timeDisplayElement.classList.remove('text-red-500', 'recording-indicator');
                updateTimeDisplay(); // Restore normal time display
            }
            
            showFeedbackToast('Recording saved');
            
            // In a real implementation, this would stop and save the recording
            // Optional: Make API call to stop server-side recording and save the file
        }
    }
    
    // Update recording duration display
    function updateRecordingDuration() {
        if (!recordingStartTime) return;
        
        const now = new Date();
        const durationMs = now - recordingStartTime;
        const seconds = Math.floor(durationMs / 1000) % 60;
        const minutes = Math.floor(durationMs / 60000) % 60;
        const hours = Math.floor(durationMs / 3600000);
        
        const formattedTime = 
            (hours ? (hours < 10 ? '0' : '') + hours + ':' : '') +
            (minutes < 10 ? '0' : '') + minutes + ':' +
            (seconds < 10 ? '0' : '') + seconds;
            
        // Find time display if not already available
        const timeDisplayElement = timeDisplay || document.querySelector('#cameraModal .text-sm:first-of-type');
        
        // Display recording duration
        if (timeDisplayElement) {
            timeDisplayElement.textContent = `REC ${formattedTime}`;
            timeDisplayElement.classList.add('text-red-500', 'recording-indicator');
        }
    }
    
    // Timeline functionality
    function initTimelineDragging() {
        // Find timeline elements if not already found
        const timeline = timelineSlider || document.querySelector('#cameraModal .relative.w-full.h-1');
        const handle = timelineHandle || document.querySelector('#cameraModal .absolute.left-\\[75\\%\\]');
        
        if (!timeline) {
            console.log("Timeline slider not found");
            return;
        }
        
        // Add timeline-slider class for styling
        timeline.classList.add('timeline-slider');
        
        if (handle) {
            // Add timeline-handle class for styling
            handle.classList.add('timeline-handle');
        }
        
        // Configure events for timeline clicking and dragging
        timeline.addEventListener('mousedown', startTimelineDrag);
        document.addEventListener('mousemove', dragTimeline);
        document.addEventListener('mouseup', stopTimelineDrag);
        
        // Direct click on timeline - jump to position
        timeline.addEventListener('click', function(e) {
            if (isDraggingTimeline) return; // Don't handle if already dragging
            
            updateTimelineFromMouse(e);
            showFeedbackToast('Seeking video...');
        });
        
        // Touch events for mobile
        timeline.addEventListener('touchstart', function(e) {
            if (e.touches && e.touches[0]) {
                const touch = e.touches[0];
                startTimelineDrag({clientX: touch.clientX, preventDefault: () => {}});
            }
        });
        
        document.addEventListener('touchmove', function(e) {
            if (!isDraggingTimeline) return;
            
            if (e.touches && e.touches[0]) {
                const touch = e.touches[0];
                dragTimeline({clientX: touch.clientX});
            }
        });
        
        document.addEventListener('touchend', stopTimelineDrag);
        
        // Start timeline updates
        startTimelineUpdates();
    }
    
    function startTimelineDrag(e) {
        if (e && e.preventDefault) e.preventDefault();
        isDraggingTimeline = true;
        updateTimelineFromMouse(e);
    }
    
    function dragTimeline(e) {
        if (!isDraggingTimeline) return;
        updateTimelineFromMouse(e);
    }
    
    function stopTimelineDrag() {
        isDraggingTimeline = false;
    }
    
    function updateTimelineFromMouse(e) {
        if (!e) return;
        
        const timeline = timelineSlider || document.querySelector('#cameraModal .relative.w-full.h-1');
        if (!timeline) return;
        
        const rect = timeline.getBoundingClientRect();
        const offsetX = e.clientX - rect.left;
        const percentage = Math.min(100, Math.max(0, (offsetX / rect.width) * 100));
        
        currentPosition = percentage;
        updateTimelinePosition();
        updateTimeDisplay();
    }
    
    function updateTimelinePosition() {
        // Find timeline elements if not cached
        const timeline = timelineSlider || document.querySelector('#cameraModal .relative.w-full.h-1');
        const handle = timelineHandle || document.querySelector('#cameraModal .absolute.left-\\[75\\%\\]');
        
        if (!timeline) return;
        
        // Update handle position
        if (handle) {
            handle.style.left = `${currentPosition}%`;
        }
        
        // Update the filled portion of the timeline
        const progressBar = timeline.querySelector('.absolute.left-0.top-0.bottom-0');
        if (progressBar) {
            progressBar.style.width = `${currentPosition}%`;
        }
    }
    
    function updateTimeDisplay() {
        // Calculate time based on position and sample time range
        // In real implementation, this would use the actual video time
        const startTime = new Date();
        startTime.setHours(15, 0, 0, 0); // 15:00:00
        
        const totalDurationMs = 2 * 60 * 60 * 1000; // 2 hours in milliseconds
        const currentTimeMs = (currentPosition / 100) * totalDurationMs;
        
        const currentTime = new Date(startTime.getTime() + currentTimeMs);
        const formattedTime = currentTime.toTimeString().split(' ')[0].substring(0, 8); // HH:MM:SS
        
        // Find time display if not cached
        const timeDisplayElement = timeDisplay || document.querySelector('#cameraModal .text-sm:first-of-type');
        
        if (timeDisplayElement && !isRecording) {
            timeDisplayElement.textContent = formattedTime;
            timeDisplayElement.classList.remove('text-red-500', 'recording-indicator');
        }
    }
    
    function startTimelineUpdates() {
        // Clear any existing interval
        if (timelineUpdateInterval) {
            clearInterval(timelineUpdateInterval);
        }
        
        // In a real implementation, this would sync with the actual video time
        timelineUpdateInterval = setInterval(() => {
            if (isPlaying && !isDraggingTimeline && !isRecording) {
                // Increment position at a rate corresponding to playback speed
                currentPosition += 0.1 * playbackSpeed;
                
                // Loop back when reaching the end
                if (currentPosition >= 100) {
                    currentPosition = 0;
                }
                
                updateTimelinePosition();
                updateTimeDisplay();
            }
        }, 100);
    }
    
    // Settings toggle functionality
    function initSettingsToggles() {
        const settingsToggles = document.querySelectorAll('#cameraModal .custom-switch input[type="checkbox"]');
        
        if (!settingsToggles || settingsToggles.length === 0) {
            console.log("No settings toggles found");
            return;
        }
        
        settingsToggles.forEach(toggle => {
            if (toggle.id === 'aiDetectionToggle') return; // Already handled separately
            
            const settingContainer = toggle.closest('.flex');
            if (!settingContainer) return;
            
            const settingLabel = settingContainer.querySelector('span');
            if (!settingLabel) return;
            
            const settingName = settingLabel.textContent.toLowerCase().replace(/\s+/g, '');
            
            toggle.addEventListener('change', function() {
                const isEnabled = this.checked;
                
                // Update state
                if (settingsStates.hasOwnProperty(settingName)) {
                    settingsStates[settingName] = isEnabled;
                }
                
                // Show feedback
                showFeedbackToast(`${settingLabel.textContent} ${isEnabled ? 'enabled' : 'disabled'}`);
                
                // Get the current camera ID
                const cameraId = window.currentModalCameraId || 
                                 document.querySelector('#modalCameraId')?.textContent?.replace(/\D/g, '') || 
                                 '1';
                
                // Send API call
                fetch(`/api/camera_setting/${cameraId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        setting: settingName,
                        enabled: isEnabled 
                    })
                })
                .then(response => {
                    if (!response.ok) throw new Error('Network response was not ok');
                    return response.json();
                })
                .then(data => {
                    console.log(`Setting '${settingName}' updated:`, data);
                })
                .catch(err => {
                    console.log('Error updating setting:', err.message);
                });
            });
        });
    }
    
    // Quality selector functionality
    function initQualitySelector() {
        // Find quality selector if not already found
        const qualityButtonElement = qualityButton || 
                                    document.querySelector('#cameraModal .flex.items-center.justify-between button') || 
                                    findElementWithText('#cameraModal button', 'High');
                                    
        if (!qualityButtonElement) {
            console.log("Quality selector not found");
            return;
        }
        
        // Create dropdown menu
        const dropdownMenu = document.createElement('div');
        dropdownMenu.className = 'quality-dropdown absolute right-0 mt-2 w-40 bg-white border border-gray-200 rounded-md shadow-lg py-1 z-10 hidden';
        
        // Add options
        qualityOptions.forEach(option => {
            const optionElement = document.createElement('button');
            optionElement.className = 'quality-option w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100';
            optionElement.textContent = option.label;
            optionElement.addEventListener('click', function() {
                selectQuality(option.label);
                dropdownMenu.classList.add('hidden');
            });
            dropdownMenu.appendChild(optionElement);
        });
        
        // Append to parent
        const container = qualityButtonElement.closest('.relative') || qualityButtonElement.parentElement;
        if (container) {
            container.style.position = 'relative';  // Ensure relative positioning for dropdown
            container.appendChild(dropdownMenu);
            
            // Toggle dropdown
            qualityButtonElement.addEventListener('click', function(e) {
                e.stopPropagation();
                dropdownMenu.classList.toggle('hidden');
            });
            
            // Close dropdown on outside click
            document.addEventListener('click', function() {
                dropdownMenu.classList.add('hidden');
            });
        }
    }
    
    function selectQuality(quality) {
        // Find quality button if not cached
        const qualityButtonElement = qualityButton || 
                                    document.querySelector('#cameraModal .flex.items-center.justify-between button') || 
                                    findElementWithText('#cameraModal button', 'High');
        
        if (!qualityButtonElement) return;
        
        // Find span element within button
        const span = qualityButtonElement.querySelector('span');
        if (span) {
            span.textContent = quality;
        } else {
            // If no span, update button text directly
            qualityButtonElement.textContent = quality;
        }
        
        currentQuality = quality;
        showFeedbackToast(`Quality changed to ${quality}`);
        
        // Get current camera ID
        const cameraId = window.currentModalCameraId ||
                         document.querySelector('#modalCameraId')?.textContent?.replace(/\D/g, '') ||
                         '1';
        
        // Send API call to update stream quality
        fetch(`/api/camera_quality/${cameraId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                quality: quality.split(' ')[0].toLowerCase()
            })
        })
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            console.log(`Camera ${cameraId} quality updated to ${quality}`);
        })
        .catch(error => {
            console.error('Error updating camera quality:', error);
        });
    }
    
    // Utility function to show toast feedback by using the one from modal.js if available
    function showFeedbackToast(message, isError = false) {
        // Check if modal.js has already created a showFeedbackToast function
        if (window.showFeedbackToast) {
            return window.showFeedbackToast(message, isError);
        }
        
        // If not, implement a simple version here
        // Create toast element if it doesn't exist
        let toast = document.getElementById('feedback-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'feedback-toast';
            toast.className = 'fixed bottom-5 left-1/2 transform -translate-x-1/2 bg-gray-800 text-white px-4 py-2 rounded-lg opacity-0 transition-opacity duration-300 z-50';
            document.body.appendChild(toast);
        }
        
        // Apply error styling if needed
        if (isError) {
            toast.classList.add('bg-red-700');
        } else {
            toast.classList.remove('bg-red-700');
        }
        
        // Update message and show toast
        toast.textContent = message;
        toast.classList.remove('opacity-0');
        
        // Hide after delay
        setTimeout(() => {
            toast.classList.add('opacity-0');
        }, 2000);
        
        return toast;
    }
    
    // Initialize all functionality
    function init() {
        // Wait a moment for DOM to be fully available
        setTimeout(() => {
            initVideoControls();
            initSettingsToggles();
            initQualitySelector();
            updateTimeDisplay(); // Initialize time display
            initFullscreenAndScreenshot(); // Add fullscreen and screenshot functionality
            
            console.log("Modal features initialized");
        }, 500);
    }
    
    // Fullscreen and Screenshot functionality
    function initFullscreenAndScreenshot() {
        // Fullscreen button
        const fullscreenButton = document.querySelector('#cameraModal button i.ri-fullscreen-line')?.closest('button');
        if (fullscreenButton) {
            fullscreenButton.addEventListener('click', function() {
                const videoContainer = document.querySelector('#modalCameraFeed').closest('.flex-1.bg-black');
                
                if (!document.fullscreenElement) {
                    // Enter fullscreen
                    if (videoContainer.requestFullscreen) {
                        videoContainer.requestFullscreen();
                    } else if (videoContainer.webkitRequestFullscreen) { // Safari
                        videoContainer.webkitRequestFullscreen();
                    } else if (videoContainer.msRequestFullscreen) { // IE/Edge
                        videoContainer.msRequestFullscreen();
                    }
                    
                    // Change icon to exit fullscreen
                    const icon = fullscreenButton.querySelector('i');
                    if (icon) {
                        icon.className = 'ri-fullscreen-exit-line';
                    }
                    
                    showFeedbackToast('Entering fullscreen mode');
                } else {
                    // Exit fullscreen
                    if (document.exitFullscreen) {
                        document.exitFullscreen();
                    } else if (document.webkitExitFullscreen) { // Safari
                        document.webkitExitFullscreen();
                    } else if (document.msExitFullscreen) { // IE/Edge
                        document.msExitFullscreen();
                    }
                    
                    // Change icon back to enter fullscreen
                    const icon = fullscreenButton.querySelector('i');
                    if (icon) {
                        icon.className = 'ri-fullscreen-line';
                    }
                    
                    showFeedbackToast('Exiting fullscreen mode');
                }
            });
            
            // Listen for fullscreen change events
            document.addEventListener('fullscreenchange', handleFullscreenChange);
            document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
            document.addEventListener('mozfullscreenchange', handleFullscreenChange);
            document.addEventListener('MSFullscreenChange', handleFullscreenChange);
        }
        
        // Screenshot button
        const screenshotButton = document.querySelector('#cameraModal button i.ri-camera-line')?.closest('button');
        if (screenshotButton) {
            screenshotButton.addEventListener('click', function() {
                takeScreenshot();
            });
        }
    }
    
    // Handle fullscreen change events
    function handleFullscreenChange() {
        const fullscreenButton = document.querySelector('#cameraModal button i.ri-fullscreen-line, #cameraModal button i.ri-fullscreen-exit-line')?.closest('button');
        if (!fullscreenButton) return;
        
        const icon = fullscreenButton.querySelector('i');
        if (!icon) return;
        
        if (document.fullscreenElement || 
            document.webkitFullscreenElement || 
            document.mozFullScreenElement || 
            document.msFullscreenElement) {
            // In fullscreen mode
            icon.className = 'ri-fullscreen-exit-line';
        } else {
            // Not in fullscreen mode
            icon.className = 'ri-fullscreen-line';
        }
    }
    
    // Take a screenshot of the current video
    function takeScreenshot() {
        const cameraFeed = document.getElementById('modalCameraFeed');
        if (!cameraFeed) {
            showFeedbackToast('Camera feed not found');
            return;
        }
        
        try {
            // Create a canvas element to draw the screenshot
            const canvas = document.createElement('canvas');
            canvas.width = cameraFeed.naturalWidth || cameraFeed.width || 1280;
            canvas.height = cameraFeed.naturalHeight || cameraFeed.height || 720;
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(cameraFeed, 0, 0, canvas.width, canvas.height);
            
            // Convert canvas to data URL
            const dataURL = canvas.toDataURL('image/png');
            
            // Create temporary anchor to trigger download
            const link = document.createElement('a');
            
            // Generate filename with camera ID and timestamp
            const cameraId = window.currentModalCameraId || 
                document.querySelector('#modalCameraId')?.textContent?.replace(/\D/g, '') || 
                'unknown';
                
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            link.download = `screenshot_camera${cameraId}_${timestamp}.png`;
            
            // Set href to data URL
            link.href = dataURL;
            
            // Append to document, click to download, then remove
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            showFeedbackToast('Screenshot saved');
        } catch (error) {
            console.error('Error taking screenshot:', error);
            showFeedbackToast('Failed to take screenshot');
        }
    }
    
    // Start initialization
    init();
    
    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        clearInterval(timelineUpdateInterval);
        clearInterval(recordingInterval);
    });
}); 