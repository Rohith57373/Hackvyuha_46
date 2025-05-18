// Camera Modal functionality
document.addEventListener('DOMContentLoaded', function() {
    // Camera Modal
    const cameraCards = document.querySelectorAll('.camera-card');
    const cameraModal = document.getElementById('cameraModal');
    const closeModal = document.getElementById('closeModal');
    const modalCameraTitle = document.getElementById('modalCameraTitle');
    const modalStatusIndicator = document.getElementById('modalStatusIndicator');
    const modalStatusText = document.getElementById('modalStatusText');
    const modalCameraId = document.getElementById('modalCameraId');
    const modalCameraLocation = document.getElementById('modalCameraLocation');
    const modalCameraFeed = document.getElementById('modalCameraFeed');
    const aiDetectionToggle = document.getElementById('aiDetectionToggle');
    
    // Track the current camera being displayed in the modal
    let currentModalCameraId = null;
    
    // Make it accessible to other scripts
    window.currentModalCameraId = null;
    
    // Debounce function to prevent excessive API calls
    function debounce(func, wait) {
        let timeout;
        return function() {
            const context = this;
            const args = arguments;
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(context, args), wait);
        };
    }
    
    // Cache for API responses to reduce repeated requests
    const apiCache = {
        cameraStatus: {},
        cacheTime: {},
        cacheTTL: 2000, // 2 seconds TTL
        
        // Get cached value or null
        get: function(key) {
            const now = Date.now();
            if (this.cacheTime[key] && now - this.cacheTime[key] < this.cacheTTL) {
                return this.cameraStatus[key];
            }
            return null;
        },
        
        // Set cache value
        set: function(key, value) {
            this.cameraStatus[key] = value;
            this.cacheTime[key] = Date.now();
        }
    };
    
    // Function to close the modal - debounced to prevent accidental multiple calls
    const closeModalFunction = debounce(function() {
        console.log("Closing modal");
        cameraModal.classList.add('opacity-0');
        
        // When closing the modal, turn off AI detection for that camera
        if (currentModalCameraId) {
            toggleCameraAIDetection(currentModalCameraId, false);
        }
        
        setTimeout(() => {
            cameraModal.classList.add('invisible');
            
            // Don't immediately stop the video - let it continue for a short time in case user reopens
            setTimeout(() => {
                // Only stop the feed if the modal is still hidden
                if (cameraModal.classList.contains('invisible')) {
                    modalCameraFeed.src = '';
                    // Reset current camera ID
                    currentModalCameraId = null;
                    window.currentModalCameraId = null;
                }
            }, 2000); // Wait 2 seconds before stopping the feed
        }, 300);
    }, 300);
    
    // Function to toggle AI detection for a specific camera
    function toggleCameraAIDetection(cameraId, enable) {
        console.log(`Toggling AI detection for camera ${cameraId} to ${enable}`);
        
        // Optimistic UI update
        apiCache.set(`ai_status_${cameraId}`, enable);
        
        // Show feedback toast
        const feedbackToast = showFeedbackToast(`${enable ? 'Enabling' : 'Disabling'} AI detection...`);
        
        fetch('/api/toggle_camera_model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                camera_id: cameraId,
                enable: enable 
            })
        })
        .then(response => {
            if (!response.ok) throw new Error('Network response was not ok');
            return response.json();
        })
        .then(data => {
            console.log(`AI detection response for camera ${cameraId}:`, data);
            
            if (!data.success) {
                throw new Error(data.error || 'Unknown error');
            }
            
            // Update feedback toast
            if (data.model_loaded) {
                updateFeedbackToast(feedbackToast, 
                    `AI detection ${data.model_enabled ? 'enabled' : 'disabled'}`);
            } else {
                updateFeedbackToast(feedbackToast, 
                    `Warning: AI model not loaded properly`, true);
            }
            
            // Cache the actual state
            apiCache.set(`ai_status_${cameraId}`, data.model_enabled);
            
            // Only refresh if there's a discrepancy and we're still displaying this camera
            if (modalCameraFeed && currentModalCameraId === cameraId && aiDetectionToggle.checked !== data.model_enabled) {
                aiDetectionToggle.checked = data.model_enabled;
            }
        })
        .catch(error => {
            console.error('Error toggling camera AI detection:', error);
            // Revert optimistic update on failure
            updateFeedbackToast(feedbackToast, 
                `Error: ${error.message || 'Failed to toggle AI detection'}`, true);
            checkCameraAIStatus(cameraId);
        });
    }
    
    // Helper function to show feedback toast
    function showFeedbackToast(message, isError = false) {
        // Create toast if it doesn't exist
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
        
        // Return toast element for later updates
        return toast;
    }
    
    // Make the function available globally
    window.showFeedbackToast = showFeedbackToast;
    
    // Helper function to update existing toast message
    function updateFeedbackToast(toast, message, isError = false) {
        if (!toast) {
            return showFeedbackToast(message, isError);
        }
        
        // Apply error styling if needed
        if (isError) {
            toast.classList.add('bg-red-700');
        } else {
            toast.classList.remove('bg-red-700');
        }
        
        // Update message
        toast.textContent = message;
        
        // Make sure it's visible
        toast.classList.remove('opacity-0');
        
        // Hide after delay
        setTimeout(() => {
            toast.classList.add('opacity-0');
        }, 2000);
        
        return toast;
    }
    
    // Check if AI detection is enabled for a specific camera
    function checkCameraAIStatus(cameraId) {
        console.log(`Checking AI status for camera ${cameraId}`);
        
        // Try to get from cache first
        const cachedStatus = apiCache.get(`ai_status_${cameraId}`);
        if (cachedStatus !== null) {
            if (aiDetectionToggle) {
                aiDetectionToggle.checked = cachedStatus;
            }
            return;
        }
        
        // If not in cache, fetch from API
        fetch(`/api/camera_model_status/${cameraId}`)
            .then(response => {
                if (!response.ok) throw new Error('Network response was not ok');
                return response.json();
            })
            .then(data => {
                console.log(`AI status for camera ${cameraId}:`, data);
                
                if ('error' in data) {
                    throw new Error(data.error || 'Unknown error');
                }
                
                if (aiDetectionToggle) {
                    // Update toggle without triggering the change event
                    aiDetectionToggle.checked = data.model_enabled;
                    
                    // Cache the result
                    apiCache.set(`ai_status_${cameraId}`, data.model_enabled);
                }
                
                // Show warning if model not loaded
                if (!data.model_loaded) {
                    showFeedbackToast('Warning: AI model not loaded properly', true);
                }
            })
            .catch(error => {
                console.error('Error checking camera AI status:', error);
                showFeedbackToast(`Error: ${error.message || 'Failed to check AI status'}`, true);
            });
    }
    
    // Open modal when clicking on a camera card - with debouncing
    const openCamera = debounce(function(card) {
        const cameraId = card.getAttribute('data-camera');
        currentModalCameraId = cameraId;
        window.currentModalCameraId = cameraId;
        
        // Get camera details from the card
        const title = card.querySelector('h3').textContent;
        const location = card.querySelector('h3 + span').textContent;
        const id = card.querySelector('.text-xs.text-gray-500').textContent;
        
        // Get camera feed URL
        const feedUrl = `/video_feed/${cameraId}`;
        
        // Get status information from the card
        let status = 'online';
        if (card.querySelector('.status-warning')) {
            status = 'warning';
        } else if (card.querySelector('.status-offline')) {
            status = 'offline';
        }
        
        // Update modal content
        modalCameraTitle.textContent = title;
        modalCameraId.textContent = id;
        modalCameraLocation.textContent = location;
        
        // Add timestamp to avoid caching
        const timestamp = new Date().getTime();
        
        // Only update source if it's changed or not already loaded
        if (!modalCameraFeed.src.includes(feedUrl) || modalCameraFeed.src === '') {
            modalCameraFeed.src = `${feedUrl}?t=${timestamp}`;
        }
        
        // Update status indicator
        modalStatusIndicator.className = 'status-indicator';
        if (status === 'online') {
            modalStatusIndicator.classList.add('status-online');
            modalStatusText.textContent = 'Live';
        } else if (status === 'warning') {
            modalStatusIndicator.classList.add('status-warning');
            modalStatusText.textContent = 'Warning';
        } else {
            modalStatusIndicator.classList.add('status-offline');
            modalStatusText.textContent = 'Offline';
        }
        
        // Check AI detection status from cache or API
        checkCameraAIStatus(cameraId);
        
        // Show modal with a fade-in animation for smoother experience
        requestAnimationFrame(() => {
            cameraModal.classList.remove('invisible');
            requestAnimationFrame(() => {
                cameraModal.classList.remove('opacity-0');
            });
        });
    }, 300); // 300ms debounce
    
    // AI Detection Toggle with optimized API call
    if (aiDetectionToggle) {
        aiDetectionToggle.addEventListener('change', debounce(function() {
            if (currentModalCameraId) {
                toggleCameraAIDetection(currentModalCameraId, this.checked);
            }
        }, 200)); // 200ms debounce
    }
    
    // Add click event listeners to camera cards
    cameraCards.forEach(card => {
        card.addEventListener('click', function() {
            openCamera(this);
        });
    });
    
    // Close modal when clicking the close button
    if (closeModal) {
        console.log("Adding event listener to close button");
        closeModal.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            closeModalFunction();
        });
    }
    
    // Close modal when clicking outside of it
    if (cameraModal) {
        cameraModal.addEventListener('click', function(event) {
            if (event.target === cameraModal) {
                closeModalFunction();
            }
        });
    }
    
    // Close modal with ESC key
    document.addEventListener('keydown', function(event) {
        if (event.key === 'Escape' && !cameraModal.classList.contains('invisible')) {
            closeModalFunction();
        }
    });
    
    // Initialize other modal buttons
    const initModalButtons = function() {
        // Play/Pause button - already handled in modal-features.js
        
        // Fullscreen button - already handled in modal-features.js
        
        // Screenshot button - already handled in modal-features.js
        
        // All settings toggles
        const settingsToggles = document.querySelectorAll('.custom-switch input[type="checkbox"]');
        settingsToggles.forEach(toggle => {
            if (toggle.id !== 'aiDetectionToggle') { // Already handled separately
                // These are now handled in modal-features.js
                // No need to add duplicate event handlers here
            }
        });
    };
    
    // Only run basic initialization for modal display/hide
    // More advanced features are handled by modal-features.js

    // Initialize buttons once DOM is loaded
    initModalButtons();
}); 