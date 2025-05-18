import cv2
import numpy as np
import torch
import time
from collections import defaultdict, deque
from datetime import datetime
import pickle
import os
from pathlib import Path

# Try required imports
try:
    from facenet_pytorch import InceptionResnetV1
    from PIL import Image
    from torchvision import transforms
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("Warning: FaceNet not available. Face recognition will not work.")

# Constants
MATCH_THRESHOLD = 0.70  # Threshold for face matching
MIN_FACE_SIZE = 40      # Minimum face size in pixels
TRACK_HISTORY = 30      # Frames to keep in memory
EMBEDDING_DIMENSION = 512  # FaceNet embedding size

class WebRTCFaceRecognition:
    def __init__(self, model_path=None):
        """Initialize face recognition system for WebRTC"""
        # Check for GPU availability and set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"WebRTC Face Recognition using device: {self.device}")
        
        # Initialize tracking states
        self.current_id = 0
        self.tracked_faces = defaultdict(dict)
        self.face_history = defaultdict(lambda: deque(maxlen=10))
        self.last_seen_embeddings = {}  # Store embeddings for persistent IDs
        
        # Face embeddings storage
        self.embedding_storage_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_embeddings')
        os.makedirs(self.embedding_storage_path, exist_ok=True)
        self.embedding_file = os.path.join(self.embedding_storage_path, 'face_embeddings.pkl')
        self.load_stored_embeddings()
        
        # Load models
        self._load_models(model_path)
        
        # Metrics
        self.metrics = {
            'total_detections': 0,
            'total_matches': 0,
            'total_new_faces': 0,
            'processing_times': deque(maxlen=100),
        }
    
    def _load_models(self, model_path):
        """Load face detection and recognition models"""
        self.face_detector = None
        self.face_encoder = None
        
        # Try to load YOLO for face detection
        try:
            from ultralytics import YOLO
            face_model_path = model_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'example', 'yolov8m-face.pt')
            if os.path.exists(face_model_path):
                self.face_detector = YOLO(face_model_path).to(self.device)
                print(f"Face detector loaded from: {face_model_path}")
            else:
                print(f"Face model not found at: {face_model_path}")
        except ImportError:
            print("YOLO not available. Face detection will not work.")
        
        # Load FaceNet for embeddings if available
        if FACENET_AVAILABLE:
            self.face_encoder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            self.transform = transforms.Compose([
                transforms.Resize(160),
                transforms.ToTensor(),
            ])
            print("FaceNet encoder loaded successfully")
        else:
            print("FaceNet not available. Face recognition will not work.")
    
    def load_stored_embeddings(self):
        """Load stored face embeddings from file"""
        try:
            if os.path.exists(self.embedding_file):
                with open(self.embedding_file, 'rb') as f:
                    stored_data = pickle.load(f)
                    self.last_seen_embeddings = stored_data.get('embeddings', {})
                    self.current_id = stored_data.get('next_id', 0)
                    print(f"Loaded {len(self.last_seen_embeddings)} face embeddings")
        except Exception as e:
            print(f"Error loading face embeddings: {e}")
            self.last_seen_embeddings = {}
    
    def save_embeddings(self):
        """Save face embeddings to file"""
        try:
            with open(self.embedding_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.last_seen_embeddings,
                    'next_id': self.current_id + 1,
                    'updated_at': datetime.now()
                }, f)
        except Exception as e:
            print(f"Error saving face embeddings: {e}")
    
    def get_face_embedding(self, face_img):
        """Convert face image to embedding vector"""
        if self.face_encoder is None:
            return None
            
        try:
            # Convert to RGB and apply preprocessing
            if face_img.shape[0] < 20 or face_img.shape[1] < 20:
                return None
                
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Apply histogram equalization for lighting normalization
            # Convert to grayscale, equalize, then back to RGB for more robust matching
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            equalized_face = cv2.equalizeHist(gray_face)
            face_img = cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2RGB)
            
            # Transform for FaceNet
            img = Image.fromarray(face_img)
            img = self.transform(img).unsqueeze(0).to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.face_encoder(img).cpu().numpy().flatten()
                
            # Normalize embedding
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            print(f"Face embedding error: {e}")
            return None
    
    def find_matching_id(self, embedding, room_id, timestamp):
        """Find matching face ID from embedding or assign new one"""
        if embedding is None:
            return None
            
        best_match_id = None
        highest_similarity = MATCH_THRESHOLD
        
        # Check against recently tracked faces first (short-term memory)
        for face_id, data in self.tracked_faces.items():
            # Skip if not seen recently
            if timestamp - data.get('last_seen', 0) > TRACK_HISTORY:
                continue
                
            # Compare with recent embeddings for this face
            for hist_embed in self.face_history[face_id]:
                similarity = np.dot(embedding, hist_embed)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_id = face_id
        
        # If no match in recent memory, check long-term stored embeddings
        if best_match_id is None:
            for stored_id, stored_data in self.last_seen_embeddings.items():
                stored_embed = stored_data.get('embedding')
                if stored_embed is not None:
                    similarity = np.dot(embedding, stored_embed)
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match_id = stored_id
        
        # Update tracking or create new ID
        if best_match_id is not None:
            # Update tracked face info
            self.tracked_faces[best_match_id] = {
                'last_seen': timestamp,
                'room_id': room_id,
            }
            
            # Update history
            self.face_history[best_match_id].append(embedding)
            
            # Update stored embedding with new one for better adaptation
            self.last_seen_embeddings[best_match_id] = {
                'embedding': embedding,
                'last_seen': timestamp,
                'room_id': room_id
            }
            
            self.metrics['total_matches'] += 1
            return best_match_id
        else:
            # Create new ID
            self.current_id += 1
            new_id = self.current_id
            
            # Initialize tracking
            self.tracked_faces[new_id] = {
                'last_seen': timestamp,
                'room_id': room_id,
            }
            
            # Add to history
            self.face_history[new_id].append(embedding)
            
            # Store for long-term memory
            self.last_seen_embeddings[new_id] = {
                'embedding': embedding,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'room_id': room_id
            }
            
            # Save to disk periodically (every 5 new faces)
            if self.current_id % 5 == 0:
                self.save_embeddings()
                
            self.metrics['total_new_faces'] += 1
            return new_id
    
    def process_frame(self, frame, room_id):
        """Process a WebRTC frame for face recognition"""
        if frame is None or self.face_detector is None:
            return frame, []
            
        start_time = time.time()
        timestamp = time.time()
        detections = []
        
        # Create a fresh copy of the frame to work with
        processed_frame = frame.copy()
        
        # Convert to RGB for YOLO
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using YOLO
        try:
            with torch.no_grad():
                # Use a very low confidence threshold to detect more faces
                results = self.face_detector(frame_rgb, conf=0.3, verbose=False)
                
            # Process detections
            for i, result in enumerate(results):
                if not hasattr(result, 'boxes') or len(result.boxes) == 0:
                    continue
                    
                for j, box in enumerate(result.boxes):
                    try:
                        # Get face coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        
                        # Skip small faces
                        if (y2 - y1) < MIN_FACE_SIZE or (x2 - x1) < MIN_FACE_SIZE:
                            continue
                        
                        # Extract face
                        face_img = processed_frame[y1:y2, x1:x2].copy()
                        
                        # Get face embedding
                        embedding = self.get_face_embedding(face_img)
                        if embedding is None:
                            continue
                            
                        # Find or assign face ID
                        face_id = self.find_matching_id(embedding, room_id, timestamp)
                        if face_id is None:
                            continue
                            
                        # Record detection metrics
                        self.metrics['total_detections'] += 1
                        
                        # Create a very obvious colored box around the face
                        # Method 1: Thick colored box
                        color = (0, 255, 0)  # Bright green for maximum visibility
                        box_thickness = 6     # Very thick box
                        
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 0), box_thickness + 4)  # Black outline
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, box_thickness)  # Green inner rectangle
                        
                        # Method 2: Add coordinates for debug
                        debug_text = f"({x1},{y1})-({x2},{y2})"
                        cv2.putText(processed_frame, debug_text, (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Add highly visible ID label
                        confidence = 0.95  # Fixed high confidence
                        conf_percentage = int(confidence * 100)
                        label = f"ID: {face_id} {conf_percentage}%"
                        
                        # Use large text with high contrast
                        font_scale = 1.2  # Very large text
                        thickness = 3     # Bold text
                        
                        # Calculate text dimensions
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        
                        # Draw a prominent label box
                        # Black background
                        cv2.rectangle(
                            processed_frame, 
                            (x1-2, y1 - text_height - 16), 
                            (x1 + text_width + 8, y1 + 4), 
                            (0, 0, 0), -1
                        )
                        
                        # Draw text in white
                        cv2.putText(
                            processed_frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness
                        )
                        
                        # Add debug indicator on each corner of the box
                        corner_size = 20
                        # Top-left corner
                        cv2.line(processed_frame, (x1, y1), (x1 + corner_size, y1), (0, 0, 255), 4)
                        cv2.line(processed_frame, (x1, y1), (x1, y1 + corner_size), (0, 0, 255), 4)
                        
                        # Top-right corner
                        cv2.line(processed_frame, (x2, y1), (x2 - corner_size, y1), (0, 0, 255), 4)
                        cv2.line(processed_frame, (x2, y1), (x2, y1 + corner_size), (0, 0, 255), 4)
                        
                        # Bottom-left corner
                        cv2.line(processed_frame, (x1, y2), (x1 + corner_size, y2), (0, 0, 255), 4)
                        cv2.line(processed_frame, (x1, y2), (x1, y2 - corner_size), (0, 0, 255), 4)
                        
                        # Bottom-right corner
                        cv2.line(processed_frame, (x2, y2), (x2 - corner_size, y2), (0, 0, 255), 4)
                        cv2.line(processed_frame, (x2, y2), (x2, y2 - corner_size), (0, 0, 255), 4)
                        
                        # Add to detections list for WebRTC
                        detections.append({
                            'id': int(face_id),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                        })
                            
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
                
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
        
        # Check if we detected any faces
        if detections:
            # Add diagnostic info - so we know detection worked
            cv2.putText(processed_frame, f"DETECTION SUCCESS: {len(detections)} faces", 
                      (10, processed_frame.shape[0] - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            # Add message if no faces detected
            cv2.putText(processed_frame, "NO FACES DETECTED", 
                      (processed_frame.shape[1]//2 - 150, processed_frame.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Add metrics and diagnostic info to frame
        processing_time = time.time() - start_time
        self.metrics['processing_times'].append(processing_time)
        fps = 1 / processing_time if processing_time > 0 else 0
        
        # Add highly visible metrics
        cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(processed_frame, f"Faces: {len(detections)}", (10, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(processed_frame, f"Total IDs: {self.current_id}", (10, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Add "Face Detection Active" indicator
        cv2.rectangle(processed_frame, (processed_frame.shape[1]-320, 10), 
                     (processed_frame.shape[1]-10, 50), (0, 100, 0), -1)
        cv2.putText(processed_frame, "FACE DETECTION ACTIVE", 
                   (processed_frame.shape[1]-310, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return processed_frame, detections
    
    def get_id_color(self, person_id):
        """Generate a consistent color for each person ID"""
        # Use hash to get consistent color for each ID
        hash_val = hash(person_id) % 360
        hsv_color = np.uint8([[[hash_val, 255, 220]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)
        return (int(bgr_color[0,0,0]), int(bgr_color[0,0,1]), int(bgr_color[0,0,2]))
    
    def add_metrics_to_frame(self, frame):
        """Add performance metrics to the frame"""
        if frame is None:
            return frame
            
        avg_time = np.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        
        y_offset = 30
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Face Detections: {self.metrics['total_detections']}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        cv2.putText(frame, f"Unique Faces: {self.current_id}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame 