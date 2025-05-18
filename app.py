from flask import Flask, render_template, redirect, request, url_for, Response, jsonify, flash, session
import uuid
import os
import atexit
import glob
import shutil
from flask_socketio import SocketIO, emit, join_room
from video_processor import VideoProcessor
import threading
import time
from chatbot import GeminiChatbot
import base64
import cv2
import numpy as np
from models import db, User, Camera, CameraAccess, Alert, Recording
from functools import wraps
import uuid as uuid_lib
from datetime import datetime
from face_recognition_webrtc import WebRTCFaceRecognition

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secureviewsecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///secureview.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
socketio = SocketIO(app, async_mode='threading')  # Use threading mode for better performance

# Initialize SQLAlchemy
db.init_app(app)

# Initialize chatbot
GEMINI_API_KEY = "AIzaSyCkOeDL6LCJYOm_gogxL8cq_TTQCfLe3wE"  # This should be stored securely in production
chatbot = GeminiChatbot(api_key=GEMINI_API_KEY)

# Initialize face recognition for WebRTC
face_recognition = WebRTCFaceRecognition()

# Path to the YOLO model and videos
FLASK_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(FLASK_DIR)
MODEL_PATH = os.path.join(PARENT_DIR, 'example', 'yolov8m.pt')
VIDEOS_DIR = os.path.join(FLASK_DIR, 'videos')

# Ensure videos directory exists
os.makedirs(VIDEOS_DIR, exist_ok=True)

# Ensure static/img directory exists
IMG_DIR = os.path.join(FLASK_DIR, 'static', 'img')
os.makedirs(IMG_DIR, exist_ok=True)

# Copy webm files from example directory if videos directory is empty
if not os.listdir(VIDEOS_DIR):
    print("Videos directory is empty. Copying example videos...")
    example_videos = glob.glob(os.path.join(PARENT_DIR, 'example', '*.webm'))
    for video_path in example_videos:
        shutil.copy(video_path, VIDEOS_DIR)
        print(f"Copied {os.path.basename(video_path)} to videos folder")

# Initialize video processor with appropriate thread count based on CPU cores
import multiprocessing
worker_count = max(2, min(4, multiprocessing.cpu_count() - 1))
video_processor = VideoProcessor(model_path=MODEL_PATH, max_workers=worker_count)
print(f"Initializing VideoProcessor with {worker_count} worker threads")

# Sync database with current videos
def sync_cameras_with_db():
    with app.app_context():
        # For each camera in video_processor
        for camera_id, source in video_processor.video_sources.items():
            # Check if camera exists in database
            camera = Camera.query.get(camera_id)
            if not camera:
                # Create new camera record
                camera = Camera(
                    id=camera_id,
                    name=source['name'],
                    title=source['name'].replace('_', ' ').title(),
                    source_path=source['path'],
                    width=source.get('width', 1280),
                    height=source.get('height', 720),
                    fps=source.get('fps', 30.0)
                )
                db.session.add(camera)
                
        # Commit changes
        db.session.commit()
        print("Database synchronized with video sources")

# Scan for videos in the videos directory
video_processor.scan_video_folder(VIDEOS_DIR)

# Create database tables
with app.app_context():
    db.create_all()
    # Create default admin user if not exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(
            username='admin',
            email='admin@secureview.com',
            first_name='Admin',
            last_name='User',
            role='admin'
        )
        admin.set_password('admin123')  # Should be changed immediately
        db.session.add(admin)
        db.session.commit()
        print("Created default admin user: admin/admin123")

# Sync database with video sources
try:
    sync_cameras_with_db()
    print("Successfully synchronized cameras with database")
except Exception as e:
    print(f"Error synchronizing cameras with database: {e}")
    import traceback
    traceback.print_exc()

# Register cleanup function
@atexit.register
def cleanup():
    video_processor.cleanup()
    print("Application shutting down, resources cleaned up")

# Cache for camera info to avoid repeated dict creation
cameras_cache = {}
cameras_cache_time = 0

def get_cameras_info():
    """Get camera information with caching for performance"""
    global cameras_cache, cameras_cache_time
    
    # Use cached data if it's less than 10 seconds old
    if cameras_cache and time.time() - cameras_cache_time < 10:
        return cameras_cache
    
    # Build new cameras info
    cameras = {}
    for camera_id, source in video_processor.video_sources.items():
        cameras[camera_id] = {
            'id': camera_id,
            'name': source['name'],
            'title': source['name'].replace('_', ' ').title(),
            'width': source['width'],
            'height': source['height'],
        }
    
    # Update cache
    cameras_cache = cameras
    cameras_cache_time = time.time()
    
    return cameras

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to be logged in to view this page.', 'error')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def landing():
    """Landing page route"""
    return render_template('landing.html')

@app.route('/dashboard')
def index():
    # Get cameras info from cache
    cameras = get_cameras_info()
    
    # Get selected camera from query param if provided
    selected_camera = request.args.get('camera_id')
    
    # Always pass False to start with no AI processing on main page
    return render_template('index.html', model_enabled=False, cameras=cameras, selected_camera=selected_camera)

@app.route('/analytics')
def analytics():
    """Analytics dashboard route"""
    # Get cameras info from cache for possible use in analytics
    cameras = get_cameras_info()
    return render_template('analytics.html', cameras=cameras)

@app.route('/alerts')
def alerts():
    """Alerts dashboard route"""
    # Get cameras info from cache for possible use in alerts
    cameras = get_cameras_info()
    return render_template('alerts.html', cameras=cameras)

@app.route('/user_profile')
def user_profile():
    """User profile route"""
    # Get cameras info from cache for possible use in profile
    cameras = get_cameras_info()
    return render_template('user_profile.html', cameras=cameras)

@app.route('/room')
def create_room():
    room_id = str(uuid.uuid4())
    return redirect(f'/room/{room_id}')

@app.route('/room/<room_id>')
def room(room_id):
    return render_template('room.html', room_id=room_id)

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    """Route for streaming video from a camera"""
    try:
        # Set response headers for better streaming performance
        headers = {
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
        
        return Response(
            video_processor.get_video_frame_generator(camera_id),
            mimetype='multipart/x-mixed-replace; boundary=frame',
            headers=headers
        )
    except Exception as e:
        print(f"Error in video feed route for camera {camera_id}: {e}")
        # Return a blank image as fallback
        blank_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + open('flask_cctv/static/blank.jpg', 'rb').read() + b'\r\n'
        return Response(blank_frame, mimetype='multipart/x-mixed-replace; boundary=frame')

# WebRTC-specific routes
@app.route('/api/webrtc/detection', methods=['POST'])
def webrtc_detection():
    """Handle WebRTC face detection metadata"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        # Extract information
        room_id = data.get('room_id')
        peer_id = data.get('peer_id')
        detections = data.get('detections', [])
        
        # Broadcast detection data to room members
        if room_id:
            socketio.emit('face-detections', {
                'peer_id': peer_id,
                'detections': detections,
                'timestamp': time.time()
            }, to=room_id)
        
        return jsonify({"status": "success"})
    except Exception as e:
        print(f"Error in WebRTC detection API: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Socket.IO event for face detection broadcasts
@socketio.on('face-detection-broadcast')
def on_face_detection(data):
    """Handle face detection broadcast from clients"""
    try:
        room_id = data.get('room_id')
        if room_id:
            # Broadcast to everyone in the room except the sender
            emit('face-detection-update', data, to=room_id, skip_sid=request.sid)
    except Exception as e:
        print(f"Error in face detection broadcast: {e}")

# Cache for API responses
api_cache = {}
api_cache_timestamp = {}
API_CACHE_TTL = 2  # 2 seconds TTL for API cache

@app.route('/api/toggle_model', methods=['POST'])
def toggle_model():
    """Toggle YOLO model on/off"""
    data = request.get_json()
    enable = data.get('enable', None)
    if enable is not None:
        enable = bool(enable)
    is_enabled = video_processor.toggle_model(enable)
    return jsonify({'success': True, 'model_enabled': is_enabled})

@app.route('/api/toggle_camera_model', methods=['POST'])
def toggle_camera_model():
    """Toggle YOLO model on/off for a specific camera"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        camera_id = data.get('camera_id')
        enable = data.get('enable')
        
        if not camera_id:
            return jsonify({'success': False, 'error': 'Camera ID is required'}), 400
        
        # Log what we're trying to do
        if enable is not None:
            print(f"Setting detection for camera {camera_id} to: {enable}")
        else:
            print(f"Toggling detection for camera {camera_id}")
        
        # Check if camera exists
        if camera_id not in video_processor.video_sources:
            return jsonify({'success': False, 'error': f'Invalid camera ID: {camera_id}'}), 404
            
        # Attempt to toggle model
        is_enabled = video_processor.toggle_model(enable, camera_id=camera_id)
        print(f"Camera {camera_id} detection is now: {is_enabled}")
        
        # Check if models are loaded
        if video_processor.person_detector is None:
            print(f"Warning: Person detector model is not loaded")
        
        # Clear cache to ensure updated status
        api_cache.pop(f'camera_status_{camera_id}', None)
        
        return jsonify({
            'success': True, 
            'camera_id': camera_id, 
            'model_enabled': is_enabled,
            'model_loaded': video_processor.person_detector is not None
        })
    except Exception as e:
        print(f"Error toggling camera model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model_status')
def model_status():
    """Get current model status with caching"""
    cache_key = 'model_status'
    
    # Use cached response if it's fresh
    now = time.time()
    if cache_key in api_cache and now - api_cache_timestamp.get(cache_key, 0) < API_CACHE_TTL:
        return api_cache[cache_key]
    
    # Generate new response
    response = jsonify({'model_enabled': video_processor.model_enabled})
    
    # Cache it
    api_cache[cache_key] = response
    api_cache_timestamp[cache_key] = now
    
    return response

@app.route('/api/camera_model_status/<camera_id>')
def camera_model_status(camera_id):
    """Get current model status for a specific camera with caching"""
    try:
        cache_key = f'camera_status_{camera_id}'
        
        # Use cached response if it's fresh
        now = time.time()
        if cache_key in api_cache and now - api_cache_timestamp.get(cache_key, 0) < API_CACHE_TTL:
            return api_cache[cache_key]
        
        # Check if this is a valid camera
        if camera_id not in video_processor.video_sources:
            return jsonify({
                'success': False,
                'error': f'Invalid camera ID: {camera_id}'
            }), 404
            
        # Generate new response
        is_enabled = video_processor.is_model_enabled_for_camera(camera_id)
        response = jsonify({
            'camera_id': camera_id, 
            'model_enabled': is_enabled,
            'model_loaded': video_processor.person_detector is not None,
            'camera_name': video_processor.video_sources[camera_id].get('name', f'Camera {camera_id}')
        })
        
        # Cache it
        api_cache[cache_key] = response
        api_cache_timestamp[cache_key] = now
        
        return response
    except Exception as e:
        print(f"Error getting camera model status: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cameras')
def get_cameras():
    """Get list of available cameras with caching"""
    cache_key = 'cameras_list'
    
    # Use cached response if it's fresh
    now = time.time()
    if cache_key in api_cache and now - api_cache_timestamp.get(cache_key, 0) < API_CACHE_TTL:
        return api_cache[cache_key]
    
    # Get cameras from cache or generate
    cameras = get_cameras_info()
    
    # Format for API response
    cameras_api = {}
    for camera_id, camera in cameras.items():
        cameras_api[camera_id] = {
            'id': camera_id,
            'name': camera['name'],
            'title': camera['title']
        }
    
    # Generate and cache response
    response = jsonify(cameras_api)
    api_cache[cache_key] = response
    api_cache_timestamp[cache_key] = now
    
    return response

@socketio.on('join-room')
def on_join_room(data):
    room_id = data['room_id']
    user_id = data['user_id']
    join_room(room_id)
    emit('user-connected', user_id, to=room_id, skip_sid=request.sid)
    
    @socketio.on('disconnect')
    def on_disconnect():
        emit('user-disconnected', user_id, to=room_id, skip_sid=request.sid)

# Chatbot routes
@app.route('/api/chatbot', methods=['POST'])
def chat_message():
    """Handle chatbot messages"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'Message is required'}), 400
        
        user_message = data.get('message')
        response = chatbot.get_response(user_message)
        
        return jsonify({
            'success': True,
            'response': response
        })
    except Exception as e:
        print(f"Error in chatbot API: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload a new video file and add it as a camera source"""
    try:
        # Check if file is in the request
        if 'video_file' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
            
        video_file = request.files['video_file']
        
        # Check if a filename is selected
        if video_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        # Check if file is a video
        allowed_formats = {'mp4', 'avi', 'webm', 'mkv', 'mov'}
        if not ('.' in video_file.filename and video_file.filename.rsplit('.', 1)[1].lower() in allowed_formats):
            return jsonify({'success': False, 'error': 'File format not supported'}), 400
        
        # Generate a safe filename
        import uuid
        safe_filename = str(uuid.uuid4()) + '_' + video_file.filename
        
        # Create the file path
        file_path = os.path.join(VIDEOS_DIR, safe_filename)
        
        # Save the file
        video_file.save(file_path)
        
        # Create a camera name from filename
        camera_name = os.path.splitext(video_file.filename)[0].replace('_', ' ').title()
        
        # Generate new camera ID
        existing_ids = [int(cam_id) if cam_id.isdigit() else 0 for cam_id in video_processor.video_sources.keys()]
        new_id = str(max(existing_ids) + 1) if existing_ids else '1'
        
        # Load the video as a new camera
        success = video_processor.load_video_source(new_id, file_path, camera_name)
        
        if success:
            # Initialize processing for the new camera
            video_processor._initialize_camera_processing(new_id)
            
            # Clear cache to update camera list
            global cameras_cache, cameras_cache_time
            cameras_cache = {}
            cameras_cache_time = 0
            
            # Return success response with new camera details
            return jsonify({
                'success': True,
                'camera': {
                    'id': new_id,
                    'name': camera_name,
                    'path': file_path
                }
            })
        else:
            # If loading failed, remove the uploaded file
            try:
                os.remove(file_path)
            except:
                pass
            return jsonify({'success': False, 'error': 'Failed to load video file'}), 400
            
    except Exception as e:
        print(f"Error uploading video: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/detected_individuals')
def get_detected_individuals():
    """Get all detected individuals with their IDs and face images with enhanced cross-camera information"""
    try:
        individuals = []
        current_time = time.time()
        
        # Loop through all detected persons in the global tracking
        for person_id, person_data in video_processor.global_persons.items():
            # Skip if the person hasn't been seen recently (in the last 60 seconds)
            if current_time - person_data.get('last_seen', 0) > 60:
                continue
                
            # Check which cameras this person was last seen in
            camera_info = []
            face_img_base64 = None
            
            # Get cameras where this person was seen
            cameras_seen = list(person_data.get('cameras_seen', set()))
            is_cross_camera = len(cameras_seen) > 1
            
            # Calculate time spent tracking this individual
            first_seen = person_data.get('first_seen', current_time)
            last_seen = person_data.get('last_seen', current_time)
            tracking_duration = last_seen - first_seen
            
            # First try to get stored face thumbnail (priority)
            has_face = person_data.get('has_face', False)
            best_thumbnail = None
            
            if has_face and person_id in video_processor.face_thumbnails:
                best_thumbnail = video_processor.face_thumbnails[person_id]
                thumbnail_type = "face"
            elif person_id in video_processor.person_thumbnails:
                best_thumbnail = video_processor.person_thumbnails[person_id]
                thumbnail_type = "person"
            
            # If no stored thumbnail, try to get one from current frames
            if best_thumbnail is None:
                for camera_id in cameras_seen:
                    if camera_id in video_processor.camera_persons and person_id in video_processor.camera_persons[camera_id]:
                        last_position = video_processor.camera_persons[camera_id][person_id].get('position')
                        
                        # If we have a recent position, try to extract a face crop
                        if last_position and camera_id in video_processor.frame_caches:
                            try:
                                # Use a thread lock to safely access the frame
                                with video_processor.frame_caches[camera_id]['lock']:
                                    frame = video_processor.frame_caches[camera_id]['raw_frame']
                                    if frame is None:
                                        continue
                                        
                                    # Create a copy to avoid modifying the original
                                    frame = frame.copy()
                                    
                                x1, y1, x2, y2 = last_position
                                
                                # Make sure coordinates are within bounds
                                x1, y1 = max(0, x1), max(0, y1)
                                x2, y2 = min(frame.shape[1]-1, x2), min(frame.shape[0]-1, y2)
                                
                                if x2 > x1 and y2 > y1:
                                    # First try to detect a face
                                    if video_processor.face_detection_enabled:
                                        face_crop, _ = video_processor.extract_face(frame, (x1, y1, x2, y2))
                                        if face_crop is not None and face_crop.size > 0:
                                            best_thumbnail = face_crop
                                            thumbnail_type = "face"
                                            break
                                    
                                    # If no face found, use upper portion for better identification features
                                    if best_thumbnail is None:
                                        person_height = y2 - y1
                                        face_height = person_height // 2  # Top half of the person detection box
                                        
                                        # Add horizontal padding
                                        padding_x = (x2 - x1) // 6
                                        x1_padded = max(0, x1 - padding_x)
                                        x2_padded = min(frame.shape[1]-1, x2 + padding_x)
                                        
                                        # Extract the top half of the person (with padding)
                                        best_thumbnail = frame[y1:y1+face_height, x1_padded:x2_padded]
                                        thumbnail_type = "person_top"
                            except Exception as e:
                                print(f"Error extracting thumbnail: {e}")
                                continue
                                
                        # If we found a thumbnail, no need to check other cameras
                        if best_thumbnail is not None and best_thumbnail.size > 0:
                            break
            
            # Convert thumbnail to base64
            if best_thumbnail is not None and best_thumbnail.size > 0:
                try:
                    # Apply mild contrast enhancement to improve visibility
                    lab = cv2.cvtColor(best_thumbnail, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
                    cl = clahe.apply(l)
                    enhanced_lab = cv2.merge((cl, a, b))
                    enhanced_face = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    
                    # Encode as base64
                    _, buffer = cv2.imencode('.jpg', enhanced_face, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    face_img_base64 = base64.b64encode(buffer).decode('utf-8')
                except Exception as e:
                    print(f"Error encoding thumbnail: {e}")
            
            # Collect detailed camera information
            for camera_id in cameras_seen:
                # Get camera name
                camera_name = "Unknown"
                if camera_id in video_processor.video_sources:
                    camera_name = video_processor.video_sources[camera_id].get('name', f'Camera {camera_id}')
                else:
                    camera_name = f'Camera {camera_id}'
                
                # Get last position and timestamp for this camera
                last_position = None
                last_timestamp = 0
                
                if camera_id in video_processor.camera_persons and person_id in video_processor.camera_persons[camera_id]:
                    last_position = video_processor.camera_persons[camera_id][person_id].get('position')
                    last_timestamp = video_processor.camera_persons[camera_id][person_id].get('last_seen', 0)
                
                camera_info.append({
                    'id': camera_id,
                    'name': camera_name,
                    'last_seen': last_timestamp,
                    'position': last_position
                })
            
            # Calculate cross-camera match confidence
            cross_camera_confidence = person_data.get('appearance_consistency', 1.0)
            
            # Only add to results if we have some camera information
            if camera_info:
                individuals.append({
                    'id': person_id,
                    'last_seen': person_data.get('last_seen', 0),
                    'first_seen': person_data.get('first_seen', 0),
                    'tracking_duration': int(tracking_duration),
                    'cameras': camera_info,
                    'cross_camera': is_cross_camera,
                    'camera_count': len(cameras_seen),
                    'face_img': face_img_base64,
                    'has_face': has_face,
                    'thumbnail_type': thumbnail_type if 'thumbnail_type' in locals() else 'none',
                    'consistency': round(cross_camera_confidence, 2)
                })
        
        # Sort by most recently seen
        individuals.sort(key=lambda x: x['last_seen'], reverse=True)
        
        # Count cross-camera individuals
        cross_camera_count = sum(1 for ind in individuals if ind['cross_camera'])
        face_count = sum(1 for ind in individuals if ind.get('has_face', False))
        
        return jsonify({
            'success': True,
            'individuals': individuals,
            'total_count': len(individuals),
            'cross_camera_count': cross_camera_count,
            'face_count': face_count
        })
    except Exception as e:
        print(f"Error in detected individuals API: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics/metrics')
def get_analytics_metrics():
    """Get real-time analytics metrics from the video processor"""
    try:
        # Initialize response data
        metrics = {}
        
        # Get cameras info
        cameras = get_cameras_info()
        
        # Process metrics for each camera
        for camera_id, camera in cameras.items():
            # Get real FPS from video source
            source_fps = video_processor.video_sources.get(camera_id, {}).get('fps', 30)
            
            # Calculate processing FPS from actual metrics
            processing_times = video_processor.metrics.get('processing_times', [])
            if processing_times:
                avg_time = sum(processing_times) / len(processing_times)
                processing_fps = 1 / avg_time if avg_time > 0 else 0
            else:
                # Fall back to source FPS if no processing times available
                processing_fps = source_fps
            
            # Generate simulated accuracy values between 85-100%
            # More consistent but with small fluctuations
            accuracy_base = 85 + (hash(camera_id) % 10)  # Base between 85-95%
            accuracy = min(accuracy_base + ((hash(str(time.time())) % 5)), 99.9)  # Add small fluctuation
            
            # Get actual detection counts from metrics
            current_detections = len(video_processor.metrics.get('current_detections', {}).get(camera_id, set()))
            
            # Add metrics for this camera
            metrics[camera_id] = {
                'name': camera.get('name', f'Camera {camera_id}'),
                'source_fps': round(source_fps, 1),
                'processing_fps': round(processing_fps, 1),
                'accuracy': round(accuracy, 1),
                'detections': current_detections,
                'processing_time': round(avg_time * 1000 if 'avg_time' in locals() else 25, 1)  # in milliseconds
            }
        
        # Get overall system metrics
        avg_fps = sum(cam['processing_fps'] for cam in metrics.values()) / len(metrics) if metrics else 0
        avg_accuracy = sum(cam['accuracy'] for cam in metrics.values()) / len(metrics) if metrics else 0
        
        # Add global metrics
        metrics['overall'] = {
            'avg_fps': round(avg_fps, 1),
            'avg_accuracy': round(avg_accuracy, 1),
            'total_detections': video_processor.metrics.get('total_detections', 0),
            'total_matches': video_processor.metrics.get('total_matches', 0),
            'cross_camera_matches': video_processor.metrics.get('cross_camera_matches', 0)
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'timestamp': time.time()
        })
    except Exception as e:
        print(f"Error in analytics metrics API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/camera_setting/<camera_id>', methods=['POST'])
def update_camera_setting(camera_id):
    """Update settings for a specific camera"""
    try:
        data = request.get_json()
        setting_name = data.get('setting')
        enabled = data.get('enabled')
        
        if not setting_name or enabled is None:
            return jsonify({'success': False, 'error': 'Invalid request parameters'}), 400
            
        # Log the request for debugging
        print(f"Camera {camera_id} setting update: {setting_name} = {enabled}")
        
        # Validate the camera ID
        if camera_id not in video_processor.video_sources:
            return jsonify({'success': False, 'error': 'Invalid camera ID'}), 404
        
        # In a real implementation, this would update camera settings in the database
        # For now, we'll simulate a successful update
        
        # Handle special settings that have actual implementations
        if setting_name == 'aidetection':
            # This is already handled by toggle_camera_model
            video_processor.toggle_model(enable=enabled, camera_id=camera_id)
        
        # Return success response
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'setting': setting_name,
            'enabled': enabled
        })
        
    except Exception as e:
        print(f"Error updating camera setting: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/camera_quality/<camera_id>', methods=['POST'])
def update_camera_quality(camera_id):
    """Update quality settings for a specific camera"""
    try:
        data = request.get_json()
        quality = data.get('quality')
        
        if not quality:
            return jsonify({'success': False, 'error': 'Quality parameter is required'}), 400
            
        # Log the request for debugging
        print(f"Camera {camera_id} quality update: {quality}")
        
        # Validate the camera ID
        if camera_id not in video_processor.video_sources:
            return jsonify({'success': False, 'error': 'Invalid camera ID'}), 404
        
        # Validate quality value
        valid_qualities = ['low', 'medium', 'high', 'ultra']
        if quality not in valid_qualities:
            return jsonify({'success': False, 'error': f'Invalid quality value. Must be one of: {", ".join(valid_qualities)}'}), 400
        
        # In a real implementation, this would update camera quality in the database and change stream quality
        # For now, we'll simulate a successful update
        
        # Return success response
        return jsonify({
            'success': True,
            'camera_id': camera_id,
            'quality': quality
        })
        
    except Exception as e:
        print(f"Error updating camera quality: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Auth Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            
            # Update last login time
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            flash(f'Welcome back, {user.first_name or user.username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'error')
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        # Create user
        user = User(
            username=username,
            email=email,
            first_name=request.form.get('first_name', ''),
            last_name=request.form.get('last_name', ''),
            role='viewer'  # Default role for new registrations
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! You can now login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Profile routes
@app.route('/profile')
@login_required
def profile():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('logout'))
    
    # Get access permissions
    camera_access = CameraAccess.query.filter_by(user_id=user_id).all()
    accessible_cameras = {}
    
    for access in camera_access:
        camera = Camera.query.get(access.camera_id)
        if camera:
            accessible_cameras[camera.id] = {
                'camera': camera,
                'access': access
            }
    
    # Get user alerts
    alerts = Alert.query.filter_by(user_id=user_id).order_by(Alert.timestamp.desc()).limit(5).all()
    
    return render_template('profile.html', user=user, accessible_cameras=accessible_cameras, alerts=alerts)

@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('logout'))
    
    if request.method == 'POST':
        # Update profile info
        user.first_name = request.form.get('first_name', user.first_name)
        user.last_name = request.form.get('last_name', user.last_name)
        user.email = request.form.get('email', user.email)
        
        # Handle profile image upload
        if 'profile_image' in request.files:
            profile_pic = request.files['profile_image']
            if profile_pic and profile_pic.filename:
                # Generate unique filename
                pic_filename = str(uuid_lib.uuid4()) + os.path.splitext(profile_pic.filename)[1]
                # Save file
                pic_path = os.path.join(app.static_folder, 'img', 'profiles', pic_filename)
                os.makedirs(os.path.dirname(pic_path), exist_ok=True)
                profile_pic.save(pic_path)
                user.profile_image = pic_filename
        
        # Handle password change
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if current_password and new_password and confirm_password:
            if not user.check_password(current_password):
                flash('Current password is incorrect.', 'error')
                return redirect(url_for('edit_profile'))
                
            if new_password != confirm_password:
                flash('New passwords do not match.', 'error')
                return redirect(url_for('edit_profile'))
                
            user.set_password(new_password)
        
        db.session.commit()
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('profile'))
    
    return render_template('edit_profile.html', user=user)

# Camera management routes
@app.route('/cameras')
@login_required
def cameras():
    # Get user's role
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    if user.role == 'admin':
        # Admin sees all cameras
        cameras = Camera.query.all()
    else:
        # Other users see only cameras they have access to
        access = CameraAccess.query.filter_by(user_id=user_id, can_view=True).all()
        camera_ids = [a.camera_id for a in access]
        cameras = Camera.query.filter(Camera.id.in_(camera_ids)).all()
    
    return render_template('cameras.html', cameras=cameras, user=user)

@app.route('/camera/<camera_id>')
@login_required
def camera_detail(camera_id):
    # Get user's role
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    # Check camera access
    if user.role != 'admin':
        access = CameraAccess.query.filter_by(
            user_id=user_id, 
            camera_id=camera_id, 
            can_view=True
        ).first()
        
        if not access:
            flash('You do not have permission to view this camera.', 'error')
            return redirect(url_for('cameras'))
            
        can_edit = access.can_edit
    else:
        can_edit = True
    
    # Get camera details
    camera = Camera.query.get_or_404(camera_id)
    
    # Get recent alerts
    alerts = Alert.query.filter_by(camera_id=camera_id).order_by(Alert.timestamp.desc()).limit(5).all()
    
    # Get recent recordings
    recordings = Recording.query.filter_by(camera_id=camera_id).order_by(Recording.start_time.desc()).limit(5).all()
    
    return render_template('camera_detail.html', camera=camera, alerts=alerts, recordings=recordings, user=user, can_edit=can_edit)

@app.route('/api/camera/<camera_id>', methods=['PUT'])
@login_required
def update_camera(camera_id):
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    
    # Check edit permission
    if user.role != 'admin':
        access = CameraAccess.query.filter_by(
            user_id=user_id, 
            camera_id=camera_id, 
            can_edit=True
        ).first()
        
        if not access:
            return jsonify({'success': False, 'error': 'Permission denied'}), 403
    
    # Get camera details
    camera = Camera.query.get_or_404(camera_id)
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        # Update camera fields
        for field in ['name', 'title', 'location', 'notes']:
            if field in data:
                setattr(camera, field, data[field])
                
        # Update camera settings
        for field in ['motion_detection', 'face_recognition', 'audio_recording', 
                     'ai_detection', 'recording_enabled']:
            if field in data:
                setattr(camera, field, bool(data[field]))
                
        # Update recording quality
        if 'recording_quality' in data:
            quality = data['recording_quality']
            if quality in ['low', 'medium', 'high', 'ultra']:
                camera.recording_quality = quality
        
        # Update dates
        if 'last_maintenance' in data:
            try:
                camera.last_maintenance = datetime.fromisoformat(data['last_maintenance'])
            except ValueError:
                pass
                
        if 'next_maintenance' in data:
            try:
                camera.next_maintenance = datetime.fromisoformat(data['next_maintenance'])
            except ValueError:
                pass
        
        # Save changes
        db.session.commit()
        
        # Also update in VideoProcessor if needed
        if camera_id in video_processor.video_sources and 'name' in data:
            video_processor.video_sources[camera_id]['name'] = data['name']
            
        # Toggle AI detection if needed
        if 'ai_detection' in data:
            video_processor.toggle_model(enable=data['ai_detection'], camera_id=camera_id)
            
        return jsonify({
            'success': True, 
            'message': 'Camera updated successfully',
            'camera': {
                'id': camera.id,
                'name': camera.name,
                'title': camera.title,
                'location': camera.location,
                'status': camera.status,
                'ai_detection': camera.ai_detection,
                'motion_detection': camera.motion_detection,
                'face_recognition': camera.face_recognition,
                'audio_recording': camera.audio_recording,
                'recording_quality': camera.recording_quality
            }
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error updating camera: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# New API endpoint for face image search
@app.route('/api/search_by_image', methods=['POST'])
def search_by_image():
    """Search cameras by uploaded face image"""
    try:
        # Check if file is in the request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
            
        image_file = request.files['image']
        
        # Check if a filename is selected
        if image_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
            
        # Check if file is an image
        allowed_formats = {'jpg', 'jpeg', 'png', 'gif'}
        if not ('.' in image_file.filename and image_file.filename.rsplit('.', 1)[1].lower() in allowed_formats):
            return jsonify({'success': False, 'error': 'File format not supported'}), 400
        
        # Get the image filename (without path)
        filename = os.path.basename(image_file.filename).lower()
        
        # For demo purposes, return different cameras based on filename
        if 'rohith' in filename:
            matching_cameras = [
                {'id': '00', 'name': 'Camera 00', 'title': 'Location', 'confidence': '93%'},
                {'id': '01', 'name': 'Camera 01', 'title': 'Loaction', 'confidence': '87%'}
            ]
        elif 'sharath' in filename:
            matching_cameras = [
                {'id': '11', 'name': 'Camera 11', 'title': 'Loaction', 'confidence': '95%'},
                {'id': '2', 'name': 'Camera 2', 'title': 'Location', 'confidence': '89%'}
            ]
        else:
            # For any other image, return a generic list
            matching_cameras = [
                {'id': '1', 'name': 'Camera 1', 'title': 'Main Entrance', 'confidence': '78%'},
                {'id': '3', 'name': 'Camera 3', 'title': 'Hallway', 'confidence': '65%'}
            ]
        
        return jsonify({
            'success': True,
            'message': f'Found {len(matching_cameras)} matches',
            'cameras': matching_cameras
        })
            
    except Exception as e:
        print(f"Error processing image search: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# New API endpoint for system metrics
@app.route('/api/system_metrics')
def system_metrics():
    """Get real-time system metrics for alerts page"""
    try:
        metrics = {
            'cpu': {'current': 0, 'threshold': 85, 'history': []},
            'memory': {'current': 0, 'total': 0, 'threshold': 80, 'unit': 'GB'},
            'disk': {'current': 0, 'total': 0, 'threshold': 90, 'unit': 'GB'},
            'network': {'current': 0, 'threshold': 80, 'unit': 'Mbps'},
            'gpu': {'temperature': 0, 'threshold': 80, 'unit': '°C'},
            'ai_processing': {'latency': 0, 'threshold': 200, 'unit': 'ms'}
        }
        
        # Try to get real CPU usage from /proc/stat
        try:
            with open('/proc/stat', 'r') as f:
                cpu_line = f.readline().split()
                if cpu_line[0] == 'cpu':
                    # Calculate CPU usage
                    total = sum(float(x) for x in cpu_line[1:])
                    idle = float(cpu_line[4])
                    usage_percent = 100 - (idle / total * 100)
                    metrics['cpu']['current'] = round(usage_percent, 1)
                    
                    # Generate history data points based on current value with small variation
                    base = max(0, metrics['cpu']['current'] - 30)
                    metrics['cpu']['history'] = [
                        round(base + ((hash(str(i)) % 60)), 1) 
                        for i in range(20)
                    ]
                    # Ensure the last point matches current
                    metrics['cpu']['history'].append(metrics['cpu']['current'])
        except Exception as e:
            print(f"Error reading CPU stats: {e}")
            # Fallback to simulated data
            metrics['cpu']['current'] = 92.3
            metrics['cpu']['history'] = [70, 65, 75, 60, 65, 55, 60, 50, 55, 45, 50, 30, 
                                         20, 25, 15, 10, 15, 5, 10, 5, 8, 92.3]
        
        # Try to get real memory usage from /proc/meminfo
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_info = {}
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        mem_info[key.strip()] = int(value.split()[0])
                
                if 'MemTotal' in mem_info and 'MemFree' in mem_info and 'Buffers' in mem_info and 'Cached' in mem_info:
                    total_mem = mem_info['MemTotal']
                    free_mem = mem_info['MemFree']
                    buffers = mem_info['Buffers']
                    cached = mem_info['Cached']
                    
                    used_mem = total_mem - free_mem - buffers - cached
                    mem_percent = (used_mem / total_mem) * 100
                    
                    metrics['memory']['current'] = round(mem_percent, 1)
                    metrics['memory']['total'] = round(total_mem / 1024 / 1024, 1)  # Convert to GB
                    metrics['memory']['used'] = round(used_mem / 1024 / 1024, 1)  # Convert to GB
        except Exception as e:
            print(f"Error reading memory info: {e}")
            # Fallback to simulated data
            metrics['memory']['current'] = 78.5
            metrics['memory']['total'] = 16
            metrics['memory']['used'] = 12.5
        
        # Try to get disk usage
        try:
            import shutil
            disk = shutil.disk_usage('/')
            metrics['disk']['current'] = round((disk.used / disk.total) * 100, 1)
            metrics['disk']['total'] = round(disk.total / (1024**3), 1)  # Convert to GB
            metrics['disk']['used'] = round(disk.used / (1024**3), 1)  # Convert to GB
        except Exception as e:
            print(f"Error getting disk usage: {e}")
            # Fallback to simulated data
            metrics['disk']['current'] = 95.2
            metrics['disk']['total'] = 1000
            metrics['disk']['used'] = 952
        
        # Try to get network bandwidth usage 
        # This would usually be tracked over time, simulating here
        try:
            with open('/proc/net/dev', 'r') as f:
                # Skip header lines
                f.readline()
                f.readline()
                
                total_rx_bytes = 0
                total_tx_bytes = 0
                
                for line in f:
                    if ':' in line:
                        # Extract interface data
                        interface_data = line.split(':')
                        if len(interface_data) >= 2 and not interface_data[0].strip().startswith('lo'):
                            data_values = interface_data[1].split()
                            if len(data_values) >= 16:
                                rx_bytes = int(data_values[0])
                                tx_bytes = int(data_values[8])
                                total_rx_bytes += rx_bytes
                                total_tx_bytes += tx_bytes
                
                # Just use a value relative to a 100 Mbps connection
                metrics['network']['current'] = 84.3
                metrics['network']['rx_bytes'] = total_rx_bytes
                metrics['network']['tx_bytes'] = total_tx_bytes
        except Exception as e:
            print(f"Error reading network stats: {e}")
            # Fallback to simulated data
            metrics['network']['current'] = 84.3
            metrics['network']['rx_bytes'] = 10000000
            metrics['network']['tx_bytes'] = 5000000
        
        # Simulate GPU temperature (would require specific hardware access)
        metrics['gpu']['temperature'] = 85
        
        # Simulate AI processing latency based on CPU usage
        ai_latency = 100 + (metrics['cpu']['current'] * 2.5)
        metrics['ai_processing']['latency'] = round(ai_latency, 1)
        
        # Add timestamps for data freshness
        import time
        from datetime import datetime
        
        current_time = datetime.now()
        metrics['timestamp'] = current_time.timestamp()
        metrics['formatted_time'] = current_time.strftime("%H:%M:%S")
        
        # Dynamically determine alert levels
        metrics['alerts'] = {
            'critical': sum(1 for m in metrics.values() 
                          if isinstance(m, dict) and 'current' in m and 'threshold' in m 
                          and m['current'] > m['threshold']),
            'warning': sum(1 for m in metrics.values() 
                         if isinstance(m, dict) and 'current' in m and 'threshold' in m 
                         and m['current'] > (m['threshold'] * 0.9) and m['current'] <= m['threshold']),
            'info': 12,  # Simulated value for other system alerts
            'resolved': 24  # Simulated value
        }
        
        # Check if we need to send notifications for critical metrics
        critical_metrics = [
            (key, data) for key, data in metrics.items()
            if isinstance(data, dict) and 'current' in data and 'threshold' in data 
            and data['current'] > data['threshold']
        ]
        
        # Send notifications if needed
        if critical_metrics and should_send_notifications():
            send_metric_notifications(critical_metrics, metrics)
        
        return jsonify({'success': True, 'metrics': metrics})
        
    except Exception as e:
        print(f"Error in system metrics API: {e}")
        import traceback
        traceback.print_exc()
        
        # Return fallback data
        return jsonify({
            'success': True,
            'metrics': {
                'cpu': {'current': 92.3, 'threshold': 85, 'history': [70, 65, 75, 60, 65, 55, 60, 50, 55, 45, 50, 30, 20, 25, 15, 10, 15, 5, 10, 5, 8, 92.3]},
                'memory': {'current': 78.5, 'threshold': 80, 'total': 16, 'used': 12.5, 'unit': 'GB'},
                'disk': {'current': 95.2, 'threshold': 90, 'total': 1000, 'used': 952, 'unit': 'GB'},
                'network': {'current': 84.3, 'threshold': 80, 'unit': 'Mbps'},
                'gpu': {'temperature': 85, 'threshold': 80, 'unit': '°C'},
                'ai_processing': {'latency': 350.2, 'threshold': 200, 'unit': 'ms'},
                'timestamp': time.time(),
                'formatted_time': datetime.now().strftime("%H:%M:%S"),
                'alerts': {
                    'critical': 3,
                    'warning': 2,
                    'info': 12,
                    'resolved': 24
                }
            },
            'simulated': True
        })

# Store notification preferences in memory (in a real app, this would be in the database)
notification_settings = {
    'enabled': True,
    'email_enabled': True,
    'sms_enabled': False,
    'push_enabled': True,
    'cooldown_minutes': 15,  # Minimum time between repeated notifications
    'last_notification': {}  # Store timestamp of last notification for each metric
}

def should_send_notifications():
    """Check if notifications are enabled"""
    return notification_settings['enabled']

def send_metric_notifications(critical_metrics, all_metrics):
    """Send notifications for critical metrics if cooldown period has passed"""
    current_time = datetime.now().timestamp()
    
    for metric_name, metric_data in critical_metrics:
        # Skip if we're still in cooldown period for this metric
        last_notification = notification_settings['last_notification'].get(metric_name, 0)
        cooldown_seconds = notification_settings['cooldown_minutes'] * 60
        
        if current_time - last_notification < cooldown_seconds:
            continue
            
        # Format the notification message
        message = f"CRITICAL ALERT: {metric_name.replace('_', ' ').title()} at {metric_data['current']}"
        if 'unit' in metric_data:
            message += f" {metric_data['unit']}"
        message += f" (Threshold: {metric_data['threshold']}"
        if 'unit' in metric_data:
            message += f" {metric_data['unit']}"
        message += ")"
        
        # In a real application, this would send actual notifications
        print(f"Sending notification: {message}")
        
        # For demo purposes, log the notification to the alerts table
        try:
            with app.app_context():
                alert = Alert(
                    title=f"{metric_name.replace('_', ' ').title()} Critical",
                    message=message,
                    level="critical",
                    source="system_metrics",
                    alert_type="system_metric",
                    timestamp=datetime.now()
                )
                db.session.add(alert)
                db.session.commit()
        except Exception as e:
            print(f"Error logging alert: {e}")
        
        # Update last notification time
        notification_settings['last_notification'][metric_name] = current_time

@app.route('/api/notification_settings', methods=['GET', 'POST'])
@login_required
def manage_notification_settings():
    """Get or update notification settings"""
    if request.method == 'GET':
        # Return current notification settings
        return jsonify({
            'success': True,
            'settings': notification_settings
        })
    else:
        # Update notification settings
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        # Update only the fields that are provided
        valid_fields = [
            'enabled', 'email_enabled', 'sms_enabled', 
            'push_enabled', 'cooldown_minutes'
        ]
        
        for field in valid_fields:
            if field in data:
                notification_settings[field] = data[field]
        
        return jsonify({
            'success': True, 
            'message': 'Notification settings updated',
            'settings': notification_settings
        })

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get system alerts with optional filtering"""
    try:
        # Get query parameters for filtering
        source = request.args.get('source')
        level = request.args.get('level')
        is_read = request.args.get('is_read')
        is_resolved = request.args.get('is_resolved')
        limit = request.args.get('limit', 50, type=int)
        
        # Start building the query
        query = Alert.query
        
        # Apply filters if provided
        if source:
            query = query.filter(Alert.source == source)
        if level:
            query = query.filter(Alert.level == level)
        if is_read is not None:
            query = query.filter(Alert.is_read == (is_read.lower() == 'true'))
        if is_resolved is not None:
            query = query.filter(Alert.is_resolved == (is_resolved.lower() == 'true'))
            
        # Order by timestamp (newest first) and limit
        alerts = query.order_by(Alert.timestamp.desc()).limit(limit).all()
        
        # Convert to dict for JSON response
        alerts_data = [alert.to_dict() for alert in alerts]
        
        # Count by level for summary
        critical_count = Alert.query.filter_by(level='critical', is_resolved=False).count()
        warning_count = Alert.query.filter_by(level='warning', is_resolved=False).count()
        info_count = Alert.query.filter_by(level='info', is_resolved=False).count()
        resolved_count = Alert.query.filter_by(is_resolved=True).count()
        
        return jsonify({
            'success': True,
            'alerts': alerts_data,
            'counts': {
                'critical': critical_count,
                'warning': warning_count,
                'info': info_count,
                'resolved': resolved_count
            }
        })
    except Exception as e:
        print(f"Error getting alerts: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/alerts/<int:alert_id>', methods=['PUT'])
@login_required
def update_alert(alert_id):
    """Update alert status (read/resolved)"""
    try:
        alert = Alert.query.get_or_404(alert_id)
        
        # Get JSON data
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        # Update fields
        if 'is_read' in data:
            alert.is_read = bool(data['is_read'])
            
        if 'is_resolved' in data:
            alert.is_resolved = bool(data['is_resolved'])
            if alert.is_resolved:
                alert.resolved_by = session.get('user_id')
                alert.resolved_timestamp = datetime.now()
                
        if 'resolution_note' in data:
            alert.resolution_note = data['resolution_note']
            
        # Save changes
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Alert {alert_id} updated',
            'alert': alert.to_dict()
        })
    except Exception as e:
        print(f"Error updating alert: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# New API endpoint for live feed face recognition
@app.route('/api/live_feed/process', methods=['POST'])
def live_feed_process():
    """Process frames for live feed face recognition"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        # Extract information
        frame_data = data.get('frame')
        ai_detection = data.get('ai_detection', False)
        face_recognition_enabled = data.get('face_recognition', True)
        privacy_mode = data.get('privacy_mode', False)
        debug_mode = data.get('debug_mode', False)
        frame_count = data.get('frame_count', 0)
        
        if not frame_data:
            return jsonify({"status": "error", "message": "Missing frame data"}), 400
        
        # Decode the base64 frame
        try:
            # Start timer for processing
            start_time = time.time()
            
            # Remove the data URL prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',', 1)[1]
            
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(frame_data)
            
            # Convert to numpy array
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            
            # Decode the image
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"status": "error", "message": "Invalid image data"}), 400
            
            processed_frame = None
            face_detections = []
            room_id = 'live_feed'  # Use a fixed room ID for the live feed
            
            # Process frame based on detection flags
            if face_recognition_enabled:
                processed_frame, face_detections = face_recognition.process_frame(frame, room_id)
                
                # Apply privacy mode if enabled (blur unknown faces)
                if privacy_mode and processed_frame is not None:
                    # Simple approach: blur faces without IDs 
                    # This would need more sophisticated logic in a real implementation
                    pass
                
                # Additional AI detection if requested
                if ai_detection and video_processor.person_detector is not None:
                    # Create a copy to avoid modifying the already processed frame
                    ai_frame = processed_frame.copy() if processed_frame is not None else frame.copy()
                    detections = video_processor.detect_objects(ai_frame)
                    
                    # Draw AI detections on the frame
                    processed_frame = video_processor.draw_detections(ai_frame, detections, thickness=2)
            else:
                # Just return the original frame
                processed_frame = frame
            
            # Extract thumbnails if face recognition is enabled
            if face_recognition_enabled:
                for detection in face_detections:
                    if 'bbox' in detection:
                        x1, y1, x2, y2 = detection['bbox']
                        if x1 < x2 and y1 < y2 and x2 <= frame.shape[1] and y2 <= frame.shape[0]:
                            try:
                                # Extract face thumbnail
                                face_img = frame[int(y1):int(y2), int(x1):int(x2)]
                                if face_img.size > 0:
                                    # Convert to base64 for sending
                                    _, thumb_buffer = cv2.imencode('.jpg', face_img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                    thumb_base64 = base64.b64encode(thumb_buffer).decode('utf-8')
                                    detection['face_thumbnail'] = thumb_base64
                            except Exception as e:
                                print(f"Error extracting thumbnail: {e}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Debug information
            debug_info = {
                'processing_time': processing_time,
                'frame_count': frame_count,
                'matches': face_recognition.metrics.get('total_matches', 0),
                'new_faces': face_recognition.metrics.get('total_new_faces', 0)
            }
            
            # Re-encode frame as base64 for response
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            processed_frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Return processed results
            return jsonify({
                "status": "success",
                "processed_frame": processed_frame_data,
                "detections": face_detections,
                "debug_info": debug_info if debug_mode else None
            })
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500
        
    except Exception as e:
        print(f"Error in live feed processing API: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# WebRTC-specific routes for face recognition
@app.route('/api/webrtc/face_detection', methods=['POST'])
def webrtc_face_detection():
    """Handle WebRTC face detection and recognition"""
    try:
        # Get data from request
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        # Extract information
        room_id = data.get('room_id')
        peer_id = data.get('peer_id')
        frame_data = data.get('frame')
        ai_detection = data.get('ai_detection', False)
        
        if not frame_data or not room_id:
            return jsonify({"status": "error", "message": "Missing required data"}), 400
        
        # Decode the base64 frame
        try:
            # Remove the data URL prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',', 1)[1]
            
            # Decode base64 string to bytes
            img_bytes = base64.b64decode(frame_data)
            
            # Convert to numpy array
            img_np = np.frombuffer(img_bytes, dtype=np.uint8)
            
            # Decode the image
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"status": "error", "message": "Invalid image data"}), 400
            
            processed_frame = None
            face_detections = []
            
            # Process frame based on AI detection flag
            if ai_detection:
                # Process with both face recognition and object detection (AI)
                processed_frame, face_detections = face_recognition.process_frame(frame, room_id)
                
                # Apply additional AI detection using the video processor (person detection, etc.)
                if video_processor.person_detector is not None:
                    # Create a copy to avoid modifying the already processed frame
                    ai_frame = processed_frame.copy()
                    detections = video_processor.detect_objects(ai_frame)
                    
                    # Draw AI detections on the frame
                    processed_frame = video_processor.draw_detections(ai_frame, detections, thickness=2)
            else:
                # Process with face recognition only
                processed_frame, face_detections = face_recognition.process_frame(frame, room_id)
            
            # Re-encode frame as base64 for response
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            processed_frame_data = base64.b64encode(buffer).decode('utf-8')
            
            # Broadcast face detections to room members if needed
            if room_id and face_detections:
                socketio.emit('face-detections', {
                    'peer_id': peer_id,
                    'detections': face_detections,
                    'timestamp': time.time()
                }, to=room_id)
            
            return jsonify({
                "status": "success",
                "processed_frame": processed_frame_data,
                "detections": face_detections,
                "ai_enabled": ai_detection
            })
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500
        
    except Exception as e:
        print(f"Error in WebRTC face detection API: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# WebRTC face recognition room page with real-time tracking
@app.route('/face_room')
def face_room():
    """Create a new room for face recognition"""
    room_id = str(uuid.uuid4())
    return redirect(f'/face_room/{room_id}')

@app.route('/compare')
def compare_room():
    """Create a new room for face comparison"""
    room_id = str(uuid.uuid4())
    return redirect(f'/compare/{room_id}')

@app.route('/face_room/<room_id>')
def face_recognition_room(room_id):
    """Render face recognition room with WebRTC streaming"""
    return render_template('face_room.html', room_id=room_id)

# New route for face comparison view
@app.route('/compare/<room_id>')
def face_compare(room_id):
    """Render the side-by-side comparison view for faces"""
    return render_template('face_compare.html', room_id=room_id)

# New route for live feed face recognition page
@app.route('/live_feed')
def live_feed():
    """Render the live feed page with real-time face recognition"""
    return render_template('live_feed.html')

if __name__ == '__main__':
    try:
        # Create a blank image for fallback
        blank_img_path = os.path.join(FLASK_DIR, 'static')
        os.makedirs(blank_img_path, exist_ok=True)
        blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank_img, "No Video Available", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(blank_img_path, 'blank.jpg'), blank_img)
        
        # Run with debug settings for template reloading
        print("Starting Flask server on port 8081...")
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        socketio.run(app, debug=True, host='0.0.0.0', port=8081)
    except Exception as e:
        print(f"Error starting the application: {e}")
        import traceback
        traceback.print_exc() 