// Main JavaScript for SecureView application
document.addEventListener('DOMContentLoaded', function() {
    console.log('SecureView application loaded');
    
    // Basic search functionality
    const searchInput = document.querySelector('input[placeholder="Search cameras..."]');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const searchTerm = e.target.value.toLowerCase();
            const cameraCards = document.querySelectorAll('.camera-card');
            
            cameraCards.forEach(card => {
                const title = card.querySelector('h3').textContent.toLowerCase();
                const id = card.getAttribute('data-camera');
                
                if (title.includes(searchTerm) || id.includes(searchTerm)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        });
    }

    // Navbar scroll effect
    const navbar = document.querySelector('.nav-container');
    
    // Apply scroll effect
    function handleScroll() {
        if (window.scrollY > 20) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    }

    // Listen for scroll events
    window.addEventListener('scroll', handleScroll);
    
    // Initialize on page load
    handleScroll();
    
    // Detected Individuals functionality
    const detectedIndividualsSection = document.getElementById('detectedIndividualsSection');
    const individualsList = document.getElementById('individualsList');
    const individualCount = document.getElementById('individualCount');
    const lastUpdated = document.getElementById('lastUpdated');
    const individualsLoading = document.getElementById('individualsLoading');
    const noIndividuals = document.getElementById('noIndividuals');
    
    // Check if AI detection is enabled
    let modelEnabled = false;
    let pollingInterval = null;
    
    // Check AI model status and update UI accordingly
    function checkModelStatus() {
        fetch('/api/model_status')
            .then(response => response.json())
            .then(data => {
                modelEnabled = data.model_enabled;
                
                if (modelEnabled) {
                    // Show the section and start polling for individuals
                    detectedIndividualsSection.classList.remove('hidden');
                    if (!pollingInterval) {
                        fetchDetectedIndividuals();
                        pollingInterval = setInterval(fetchDetectedIndividuals, 5000); // Poll every 5 seconds
                    }
                } else {
                    // Hide the section and stop polling
                    detectedIndividualsSection.classList.add('hidden');
                    if (pollingInterval) {
                        clearInterval(pollingInterval);
                        pollingInterval = null;
                    }
                }
            })
            .catch(error => console.error('Error checking model status:', error));
    }
    
    // Fetch detected individuals from the API
    function fetchDetectedIndividuals() {
        individualsList.classList.add('opacity-50');
        
        fetch('/api/detected_individuals')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateIndividualsDisplay(data.individuals);
                    
                    // Update count information
                    individualCount.textContent = data.total_count || data.individuals.length;
                    
                    // Update cross-camera count if available
                    const crossCameraCount = document.getElementById('crossCameraCount');
                    if (crossCameraCount) {
                        crossCameraCount.textContent = data.cross_camera_count || 
                            data.individuals.filter(ind => ind.cross_camera).length;
                    }
                    
                    // Update timestamp
                    const now = new Date();
                    lastUpdated.textContent = `Updated: ${now.toLocaleTimeString()}`;
                } else {
                    console.error('API returned error:', data.error);
                }
                
                individualsList.classList.remove('opacity-50');
                individualsLoading.classList.add('hidden');
            })
            .catch(error => {
                console.error('Error fetching individuals:', error);
                individualsList.classList.remove('opacity-50');
                individualsLoading.classList.add('hidden');
            });
    }
    
    // Update the individuals display
    function updateIndividualsDisplay(individuals) {
        // Update count
        individualCount.textContent = individuals.length;
        
        // Clear current list
        individualsList.innerHTML = '';
        
        if (individuals.length === 0) {
            noIndividuals.classList.remove('hidden');
        } else {
            noIndividuals.classList.add('hidden');
            
            // Add individual cards
            individuals.forEach(individual => {
                const card = createIndividualCard(individual);
                individualsList.appendChild(card);
            });
        }
    }
    
    // Create an individual card element
    function createIndividualCard(individual) {
        const card = document.createElement('div');
        card.className = 'bg-white border border-gray-200 rounded-lg overflow-hidden shadow-sm';
        
        // Create image container
        const imageContainer = document.createElement('div');
        imageContainer.className = 'aspect-square bg-gray-100 relative';
        
        // Individual image (face)
        if (individual.face_img) {
            const img = document.createElement('img');
            img.src = `data:image/jpeg;base64,${individual.face_img}`;
            img.alt = `Individual ${individual.id}`;
            img.className = 'w-full h-full object-cover';
            imageContainer.appendChild(img);
        } else {
            // Placeholder if no image
            const placeholder = document.createElement('div');
            placeholder.className = 'w-full h-full flex items-center justify-center';
            placeholder.innerHTML = `<i class="ri-user-line text-4xl text-gray-300"></i>`;
            imageContainer.appendChild(placeholder);
        }
        
        // Cross-camera badge - with camera count if more than 2 cameras
        if (individual.cross_camera) {
            const badge = document.createElement('div');
            const cameraCount = individual.camera_count || individual.cameras.length;
            
            badge.className = 'absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full';
            badge.textContent = cameraCount > 2 ? 
                `${cameraCount} Cameras` : 'Cross-camera';
                
            // Give it a different color if seen in 3+ cameras (more confident match)
            if (cameraCount > 2) {
                badge.className = badge.className.replace('bg-green-500', 'bg-blue-600');
            }
            
            imageContainer.appendChild(badge);
        }
        
        // Add ID badge to corner of image
        const idBadge = document.createElement('div');
        idBadge.className = 'absolute top-2 left-2 bg-blue-600 text-white text-xs font-bold px-2 py-1 rounded-full';
        idBadge.textContent = `ID: ${individual.id}`;
        imageContainer.appendChild(idBadge);
        
        card.appendChild(imageContainer);
        
        // Card content
        const content = document.createElement('div');
        content.className = 'p-3';
        
        // Create detailed header
        const header = document.createElement('div');
        header.className = 'flex justify-between items-center';
        
        // Tracking time information
        const timeInfo = document.createElement('div');
        timeInfo.className = 'flex flex-col';
        
        // Format tracking duration nicely
        const trackingDuration = individual.tracking_duration || 0;
        let durationText = '';
        
        if (trackingDuration < 60) {
            durationText = `${trackingDuration}s`;
        } else if (trackingDuration < 3600) {
            durationText = `${Math.floor(trackingDuration / 60)}m ${trackingDuration % 60}s`;
        } else {
            durationText = `${Math.floor(trackingDuration / 3600)}h ${Math.floor((trackingDuration % 3600) / 60)}m`;
        }
        
        const trackingTime = document.createElement('span');
        trackingTime.className = 'text-xs font-medium text-gray-900';
        trackingTime.textContent = `Tracked: ${durationText}`;
        timeInfo.appendChild(trackingTime);
        
        // Last seen time
        const lastSeenTime = document.createElement('span');
        lastSeenTime.className = 'text-xs text-gray-500';
        const lastSeen = new Date(individual.last_seen * 1000);
        lastSeenTime.textContent = `Last seen: ${lastSeen.toLocaleTimeString()}`;
        timeInfo.appendChild(lastSeenTime);
        
        header.appendChild(timeInfo);
        
        // Add consistency indicator with color based on score
        const consistencyScore = individual.consistency || 1.0;
        const consistencyElement = document.createElement('div');
        consistencyElement.className = 'flex flex-col items-end gap-1';
        
        // Set color based on consistency score
        let consistencyColor = 'bg-red-500';
        if (consistencyScore >= 0.9) {
            consistencyColor = 'bg-green-500';
        } else if (consistencyScore >= 0.8) {
            consistencyColor = 'bg-yellow-500';
        } else if (consistencyScore >= 0.7) {
            consistencyColor = 'bg-orange-500';
        }
        
        // Add label for match confidence
        const confidenceLabel = document.createElement('span');
        confidenceLabel.className = 'text-xs text-gray-500 text-right';
        confidenceLabel.textContent = 'Match confidence:';
        consistencyElement.appendChild(confidenceLabel);
        
        // Create progress bar with score
        const progressContainer = document.createElement('div');
        progressContainer.className = 'flex items-center gap-1';
        
        const progressBar = document.createElement('div');
        progressBar.className = 'w-16 h-2 bg-gray-200 rounded-full overflow-hidden';
        
        const progressFill = document.createElement('div');
        progressFill.className = `${consistencyColor} h-full`;
        progressFill.style.width = `${Math.min(100, consistencyScore * 100)}%`;
        
        progressBar.appendChild(progressFill);
        progressContainer.appendChild(progressBar);
        
        const scoreText = document.createElement('span');
        scoreText.className = 'text-xs font-medium';
        scoreText.textContent = `${Math.round(consistencyScore * 100)}%`;
        progressContainer.appendChild(scoreText);
        
        consistencyElement.appendChild(progressContainer);
        header.appendChild(consistencyElement);
        
        content.appendChild(header);
        
        // Camera info with icons
        const cameraInfo = document.createElement('div');
        cameraInfo.className = 'mt-3 flex flex-wrap gap-1';
        
        // Add camera badges
        individual.cameras.forEach(camera => {
            const cameraBadge = document.createElement('span');
            cameraBadge.className = 'inline-flex items-center gap-1 bg-gray-100 rounded-full px-2 py-1 text-xs';
            cameraBadge.innerHTML = `<i class="ri-cctv-fill text-gray-500"></i> ${camera.name}`;
            cameraInfo.appendChild(cameraBadge);
        });
        
        content.appendChild(cameraInfo);
        card.appendChild(content);
        
        return card;
    }
    
    // Check model status initially and then every 5 seconds
    checkModelStatus();
    setInterval(checkModelStatus, 5000);
    
    // AI Detection toggle
    const aiDetectionToggle = document.getElementById('aiDetectionToggle');
    if (aiDetectionToggle) {
        aiDetectionToggle.addEventListener('change', function() {
            fetch('/api/toggle_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ enable: this.checked }),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Model toggled:', data.model_enabled);
                
                // Update the toggle state if it's different from what we expected
                if (this.checked !== data.model_enabled) {
                    this.checked = data.model_enabled;
                }
                
                // Update UI immediately
                checkModelStatus();
            })
            .catch(error => {
                console.error('Error toggling model:', error);
                // Revert the toggle state if there was an error
                this.checked = !this.checked;
            });
        });
    }
}); 