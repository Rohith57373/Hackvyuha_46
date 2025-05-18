// Connect to socket.io server
const socket = io();

const videoGrid = document.getElementById('video-grid');
const myVideo = document.createElement('video');
myVideo.muted = true;

const peers = {};
let faceDetectionModel = null;  // Will hold the face detection model
let isDetectionEnabled = false;  // Track if AI detection is enabled
const videoCanvasMap = new Map();  // Map videos to their respective canvas elements
let detectionInterval = null;  // Interval for running detection
let isDetectionBusy = false;   // Flag to prevent overlapping detection runs

// Store face embeddings for identity matching
const knownFaces = [];
const faceCharacters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';  // Characters for face IDs

// Load face detection model
async function loadFaceDetectionModel() {
    try {
        console.log('Loading face detection model...');
        const model = await faceDetection.createDetector(
            faceDetection.SupportedModels.MediaPipeFaceDetector, 
            { 
                runtime: 'tfjs', 
                modelType: 'short',  // Use the lightweight model for performance
                maxFaces: 10,        // Limit to 10 faces for performance
                scoreThreshold: 0.75 // Increase threshold for more confident detections
            }
        );
        console.log('Face detection model loaded successfully');
        return model;
    } catch (error) {
        console.error('Error loading face detection model:', error);
        return null;
    }
}

// Initialize face detection if possible
window.addEventListener('DOMContentLoaded', async () => {
    try {
        // Check if TF libraries are loaded
        if (typeof tf !== 'undefined' && typeof faceDetection !== 'undefined') {
            // Load model in the background to not block UI
            setTimeout(async () => {
                faceDetectionModel = await loadFaceDetectionModel();
                console.log('Face detection model loaded and ready');
            }, 2000);
            
            // Add event listener to AI toggle
            const aiToggle = document.getElementById('ai-detection-toggle');
            if (aiToggle) {
                aiToggle.addEventListener('change', function() {
                    isDetectionEnabled = this.checked;
                    
                    if (isDetectionEnabled) {
                        console.log('Face detection enabled');
                        if (!detectionInterval) {
                            startFaceDetection();
                        }
                    } else {
                        console.log('Face detection disabled');
                        clearAllDetections();
                    }
                });
            }
        } else {
            console.warn('Face detection libraries not available');
        }
    } catch (error) {
        console.error('Error initializing face detection:', error);
    }
});

// PeerJS with STUN/TURN config
const myPeer = new Peer(undefined, {
    config: {
        iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            {
                urls: 'turn:numb.viagenie.ca',
                credential: 'webrtc',
                username: 'webrtc@live.com'
            }
        ]
    }
});

// Separate WebRTC connection setup from video processing
function setupVideoStream() {
    return navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 360 } },
    audio: true
    });
}

// Initialize WebRTC connections
setupVideoStream().then(stream => {
    addVideoStream(myVideo, stream);

    // Answer incoming calls
    myPeer.on('call', call => {
        console.log('Receiving call from:', call.peer);
        call.answer(stream);
        const video = document.createElement('video');
        call.on('stream', userVideoStream => {
            console.log('Received remote stream from:', call.peer);
            addVideoStream(video, userVideoStream);
        });
        call.on('error', err => {
            console.error('Call error:', err);
        });
        call.on('close', () => {
            removeVideoStream(video);
        });

        peers[call.peer] = call;
    });

    // Connect to new user after a short delay
    socket.on('user-connected', userId => {
        console.log('User connected:', userId);
        setTimeout(() => {
            connectToNewUser(userId, stream);
        }, 1000); // Delay for better sync
    });
}).catch(err => {
    console.error('Failed to get media stream:', err);
});

// Remove peer when disconnected
socket.on('user-disconnected', userId => {
    if (peers[userId]) {
        peers[userId].close();
        delete peers[userId];
    }
});

// Emit our peer ID to server
myPeer.on('open', id => {
    console.log('My peer ID:', id);
    socket.emit('join-room', { room_id: ROOM_ID, user_id: id });
});

// Make call to new user
function connectToNewUser(userId, stream) {
    if (peers[userId]) return; // Avoid double calling
    const call = myPeer.call(userId, stream);
    const video = document.createElement('video');

    call.on('stream', userVideoStream => {
        addVideoStream(video, userVideoStream);
    });
    call.on('error', err => {
        console.error('Call error:', err);
    });
    call.on('close', () => {
        removeVideoStream(video);
    });

    peers[userId] = call;
}

// Function to remove video stream and its associated elements
function removeVideoStream(video) {
    if (videoCanvasMap.has(video)) {
        const container = video.parentElement;
        if (container) {
            container.remove();
        }
        videoCanvasMap.delete(video);
    } else {
        video.remove();
    }
    updateParticipantCount();
}

// Utility to add stream to grid with detection support
function addVideoStream(video, stream) {
    video.srcObject = stream;
    video.addEventListener('loadedmetadata', () => {
        video.play();
    });
    
    // Create container for video and overlay canvas
    const videoContainer = document.createElement('div');
    videoContainer.className = 'video-container';
    
    // Create canvas for drawing detections
    const canvas = document.createElement('canvas');
    canvas.className = 'detection-canvas';
    
    // Create detection info element
    const detectionInfo = document.createElement('div');
    detectionInfo.className = 'detection-container';
    detectionInfo.style.display = 'none';
    
    // Add video to container
    videoContainer.appendChild(video);
    videoContainer.appendChild(canvas);
    videoContainer.appendChild(detectionInfo);
    
    // Add container to grid
    videoGrid.appendChild(videoContainer);
    
    // Map video to its canvas for face detection
    videoCanvasMap.set(video, { canvas, detectionInfo });
    
    // If detection is already enabled, apply it to this new video
    if (isDetectionEnabled && faceDetectionModel) {
        // Run initial detection after a short delay to ensure video is ready
        setTimeout(() => {
            detectFaces(video, canvas, detectionInfo);
        }, 1000);
    }
    
    updateParticipantCount();
}

// Generate a face embedding for matching
function getFaceEmbedding(face) {
    // Simple method: use face landmarks as an "embedding" for matching
    // In a real app, you'd use a proper face recognition model
    if (!face.keypoints || face.keypoints.length === 0) {
        // If no landmarks, use box coordinates and dimensions
        return [
            face.box.xMin, 
            face.box.yMin, 
            face.box.width, 
            face.box.height
        ];
    }
    
    // Use landmark positions as a simple embedding
    return face.keypoints.flatMap(kp => [kp.x, kp.y]);
}

// Check if a face matches any known face
function matchFaceToKnown(faceEmbedding) {
    if (knownFaces.length === 0) return -1;
    
    let bestMatchIndex = -1;
    let bestMatchScore = Number.MAX_VALUE;
    
    // Simple Euclidean distance between embeddings
    for (let i = 0; i < knownFaces.length; i++) {
        const knownEmbedding = knownFaces[i].embedding;
        
        // Skip if embeddings are incompatible lengths
        if (!knownEmbedding || !faceEmbedding || 
            knownEmbedding.length === 0 || faceEmbedding.length === 0) {
            continue;
        }
        
        // Use the minimum length between embeddings for comparison
        const minLength = Math.min(faceEmbedding.length, knownEmbedding.length);
        if (minLength === 0) continue;
        
        // Calculate distance between embeddings
        let distanceSum = 0;
        for (let j = 0; j < minLength; j++) {
            distanceSum += Math.pow(faceEmbedding[j] - knownEmbedding[j], 2);
        }
        const distance = Math.sqrt(distanceSum);
        
        // If distance is below threshold, consider it a match
        if (distance < 100 && distance < bestMatchScore) {
            bestMatchScore = distance;
            bestMatchIndex = i;
        }
    }
    
    return bestMatchIndex;
}

// Enhanced face detection processing with consistent character IDs
async function detectFaces(video, canvas, detectionInfo) {
    if (!faceDetectionModel || !isDetectionEnabled) return;
    
    try {
        // Ensure the video is playing and has dimensions
        if (video.readyState !== 4 || video.paused || video.style.display === 'none') {
            return;
        }
        
        // Match canvas size and position exactly to video
        const videoRect = video.getBoundingClientRect();
        canvas.width = video.videoWidth || videoRect.width;
        canvas.height = video.videoHeight || videoRect.height;
        
        // Ensure canvas is correctly positioned over video
        canvas.style.position = 'absolute';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        
        // Get detection results
        const predictions = await faceDetectionModel.estimateFaces(video);
        
        // Clear previous drawings
        const ctx = canvas.getContext('2d', { alpha: true });
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // If faces detected, draw enhanced bounding boxes with character IDs
        if (predictions && predictions.length > 0) {
            // Show detection info
            detectionInfo.textContent = `Faces: ${predictions.length}`;
            detectionInfo.style.display = 'block';
            
            // Get scaling factors to account for video-to-display size differences
            const scaleX = canvas.width / (video.videoWidth || videoRect.width);
            const scaleY = canvas.height / (video.videoHeight || videoRect.height);
            
            // Generate random colors for each unique face (consistent per character ID)
            const colorMap = {};
            
            // Draw each face detection with enhanced visual style and character IDs
            const processedFaces = predictions.map(prediction => {
                // Create embedding for this face
                const embedding = getFaceEmbedding(prediction);
                
                // Try to match with known faces
                const matchIndex = matchFaceToKnown(embedding);
                
                let faceID;
                let color;
                
                if (matchIndex >= 0) {
                    // Face matched with known face
                    faceID = knownFaces[matchIndex].id;
                    color = knownFaces[matchIndex].color;
                    
                    // Update embedding with latest
                    knownFaces[matchIndex].embedding = embedding;
                    knownFaces[matchIndex].lastSeen = Date.now();
                } else {
                    // New face
                    const newCharIndex = knownFaces.length % faceCharacters.length;
                    faceID = faceCharacters[newCharIndex];
                    
                    // Generate vibrant color for this face
                    color = `hsl(${(newCharIndex * 137) % 360}, 80%, 60%)`;
                    
                    // Add to known faces
                    knownFaces.push({
                        id: faceID,
                        embedding: embedding,
                        color: color,
                        lastSeen: Date.now()
                    });
                }
                
                colorMap[faceID] = color;
                
                return {
                    ...prediction,
                    faceID,
                    color
                };
            });
            
            // Draw all faces with their assigned IDs and colors
            processedFaces.forEach(face => {
                const box = face.box;
                const startX = box.xMin * scaleX;
                const startY = box.yMin * scaleY;
                const width = box.width * scaleX;
                const height = box.height * scaleY;
                const color = face.color;
                
                // Draw fancy bounding box with thick border
                ctx.lineWidth = 3;
                ctx.strokeStyle = color;
                ctx.strokeRect(startX, startY, width, height);
                
                // Add a lighter glowing effect to avoid performance issues
                ctx.shadowColor = color;
                ctx.shadowBlur = 10;
                ctx.strokeRect(startX, startY, width, height);
                ctx.shadowBlur = 0;
                
                // Draw semi-transparent background for label
                ctx.fillStyle = `${color}aa`;  // More transparency
                const labelWidth = 80;
                const labelHeight = 24;
                ctx.fillRect(startX, startY - labelHeight, labelWidth, labelHeight);
                
                // Draw face ID and confidence
                ctx.fillStyle = '#FFFFFF';
                ctx.font = 'bold 14px Inter';  // Slightly smaller font
                const confidence = Math.round(face.score * 100);
                ctx.fillText(`ID ${face.faceID}: ${confidence}%`, startX + 5, startY - 8);
                
                // Draw face landmarks if available (only key points)
                if (face.keypoints && face.keypoints.length > 0) {
                    // Limit to key landmarks (eyes, nose) for performance
                    const keyLandmarks = [0, 1, 2]; // Indices for important landmarks
                    keyLandmarks.forEach(idx => {
                        if (face.keypoints[idx]) {
                            const keypoint = face.keypoints[idx];
                            const x = keypoint.x * scaleX;
                            const y = keypoint.y * scaleY;
                            
                            // Draw landmark point
                            ctx.beginPath();
                            ctx.arc(x, y, 2, 0, 2 * Math.PI);
                            ctx.fillStyle = '#FFFFFF';
                            ctx.fill();
                        }
                    });
                }
            });
            
            // Draw info box showing total count in corner
            ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
            ctx.fillRect(10, 10, 100, 24);
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '12px Inter';
            ctx.fillText(`Total Faces: ${predictions.length}`, 15, 25);
            
            // Prune old faces that haven't been seen in a while (15 seconds)
            const now = Date.now();
            for (let i = knownFaces.length - 1; i >= 0; i--) {
                if (now - knownFaces[i].lastSeen > 15000) {
                    knownFaces.splice(i, 1);
                }
            }
        } else {
            // Hide detection info if no faces
            detectionInfo.style.display = 'none';
        }
    } catch (error) {
        console.error('Error during face detection:', error);
        detectionInfo.style.display = 'none';
    }
}

// Run detection on all videos with throttling
async function runDetectionOnAllVideos() {
    if (!isDetectionEnabled || !faceDetectionModel || isDetectionBusy) return;
    
    isDetectionBusy = true;
    try {
        const promises = [];
        videoCanvasMap.forEach((elements, video) => {
            promises.push(detectFaces(video, elements.canvas, elements.detectionInfo));
        });
        
        await Promise.all(promises);
    } catch (error) {
        console.error("Error running face detection:", error);
    } finally {
        isDetectionBusy = false;
    }
}

// Start face detection interval with adaptive rate
function startFaceDetection() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
    }
    
    // Use requestAnimationFrame for smoother performance
    let lastFrameTime = 0;
    const detectionRate = 150; // ms between detection runs (lower = more CPU usage)
    
    function detectFrame(timestamp) {
        if (!isDetectionEnabled) return;
        
        const elapsed = timestamp - lastFrameTime;
        if (elapsed > detectionRate) {
            lastFrameTime = timestamp;
            runDetectionOnAllVideos();
        }
        
        requestAnimationFrame(detectFrame);
    }
    
    requestAnimationFrame(detectFrame);
}

// Clear all detections
function clearAllDetections() {
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
    
    videoCanvasMap.forEach((elements, video) => {
        const ctx = elements.canvas.getContext('2d');
        ctx.clearRect(0, 0, elements.canvas.width, elements.canvas.height);
        elements.detectionInfo.style.display = 'none';
    });
    
    isDetectionEnabled = false;
}

// Add participant counter
function updateParticipantCount() {
    const count = document.querySelectorAll('.video-container').length;
    const countElement = document.getElementById('participant-count');
    if (countElement) {
        countElement.innerText = count;
    } else {
        const header = document.querySelector('header .container');
        if (header) {
        const countDiv = document.createElement('div');
        countDiv.classList.add('flex', 'items-center', 'gap-2', 'text-sm', 'bg-gray-100', 'px-3', 'py-1.5', 'rounded-full');
        countDiv.innerHTML = `
            <span class="text-gray-600">Participants:</span>
            <span id="participant-count" class="font-medium text-gray-800">${count}</span>
        `;
        header.appendChild(countDiv);
    }
}
}

// Socket event for receiving remote face detection results
socket.on('face-detections', (data) => {
    // Handle broadcast face detection data from other peers
    // This could be implemented to optimize performance by sharing detection results
    console.log('Received face detection data:', data);
});

// Clean up resources when page unloads
window.addEventListener('unload', () => {
    clearAllDetections();
    
    // Close all peer connections
    for (let id in peers) {
        peers[id].close();
    }
});

// Handle window resize to adjust canvas positions
window.addEventListener('resize', () => {
    if (isDetectionEnabled) {
        // Force refresh of canvas positions
        videoCanvasMap.forEach((elements, video) => {
            const videoRect = video.getBoundingClientRect();
            const canvas = elements.canvas;
            
            canvas.width = video.videoWidth || videoRect.width;
            canvas.height = video.videoHeight || videoRect.height;
            canvas.style.width = '100%';
            canvas.style.height = '100%';
        });
    }
}); 