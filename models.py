from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize SQLAlchemy
db = SQLAlchemy()

class User(db.Model):
    """User model for authentication and profile information"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(64))
    last_name = db.Column(db.String(64))
    role = db.Column(db.String(20), default='operator')  # admin, operator, viewer
    is_active = db.Column(db.Boolean, default=True)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    profile_image = db.Column(db.String(256), default='default.png')
    
    # Relationships
    camera_access = db.relationship('CameraAccess', back_populates='user', lazy='dynamic')
    alerts = db.relationship('Alert', back_populates='user', foreign_keys='Alert.user_id', lazy='dynamic')
    resolved_alerts = db.relationship('Alert', foreign_keys='Alert.resolved_by', lazy='dynamic')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
        
    def __repr__(self):
        return f'<User {self.username}>'

class Camera(db.Model):
    """Camera model for storing camera information"""
    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    location = db.Column(db.String(120))
    source_path = db.Column(db.String(256), nullable=False)
    width = db.Column(db.Integer, default=1280)
    height = db.Column(db.Integer, default=720)
    fps = db.Column(db.Float, default=30.0)
    status = db.Column(db.String(20), default='online')  # online, offline, warning
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Camera settings
    motion_detection = db.Column(db.Boolean, default=True)
    face_recognition = db.Column(db.Boolean, default=True)
    audio_recording = db.Column(db.Boolean, default=False)
    ai_detection = db.Column(db.Boolean, default=True)
    recording_enabled = db.Column(db.Boolean, default=True)
    recording_quality = db.Column(db.String(20), default='high')  # low, medium, high, ultra
    
    # Maintenance info
    last_maintenance = db.Column(db.DateTime)
    next_maintenance = db.Column(db.DateTime)
    notes = db.Column(db.Text)
    
    # Relationships
    access_rules = db.relationship('CameraAccess', back_populates='camera', lazy='dynamic')
    alerts = db.relationship('Alert', back_populates='camera', lazy='dynamic')
    recordings = db.relationship('Recording', back_populates='camera', lazy='dynamic')
    
    def __repr__(self):
        return f'<Camera {self.id}: {self.name}>'

class CameraAccess(db.Model):
    """Model for camera access control"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    camera_id = db.Column(db.String(36), db.ForeignKey('camera.id'), nullable=False)
    can_view = db.Column(db.Boolean, default=True)
    can_control = db.Column(db.Boolean, default=False)
    can_edit = db.Column(db.Boolean, default=False)
    can_download = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', back_populates='camera_access')
    camera = db.relationship('Camera', back_populates='access_rules')
    
    def __repr__(self):
        return f'<CameraAccess User:{self.user_id} Camera:{self.camera_id}>'

class Alert(db.Model):
    """Model for camera alerts"""
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.String(36), db.ForeignKey('camera.id'), nullable=True)  # Can be null for system alerts
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # Can be null for system alerts
    title = db.Column(db.String(120), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # motion, face, person, system_metric
    source = db.Column(db.String(50), default='camera')  # camera, system_metrics, security, etc.
    level = db.Column(db.String(20), default='info')  # info, warning, critical
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    is_resolved = db.Column(db.Boolean, default=False)
    resolved_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    resolution_note = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    resolved_timestamp = db.Column(db.DateTime, nullable=True)
    thumbnail_path = db.Column(db.String(256), nullable=True)
    metric_data = db.Column(db.Text, nullable=True)  # JSON data with metric values
    
    # Relationships
    camera = db.relationship('Camera', back_populates='alerts')
    user = db.relationship('User', back_populates='alerts', foreign_keys=[user_id])
    resolver = db.relationship('User', foreign_keys=[resolved_by])
    
    def __repr__(self):
        if self.camera_id:
            return f'<Alert {self.id}: {self.alert_type} on Camera {self.camera_id}>'
        else:
            return f'<Alert {self.id}: {self.source} - {self.level}>'
            
    def to_dict(self):
        """Convert alert to dictionary for API responses"""
        return {
            'id': self.id,
            'title': self.title,
            'camera_id': self.camera_id,
            'alert_type': self.alert_type,
            'source': self.source,
            'level': self.level,
            'message': self.message,
            'is_read': self.is_read,
            'is_resolved': self.is_resolved,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'resolved_timestamp': self.resolved_timestamp.isoformat() if self.resolved_timestamp else None
        }

class Recording(db.Model):
    """Model for camera recordings"""
    id = db.Column(db.Integer, primary_key=True)
    camera_id = db.Column(db.String(36), db.ForeignKey('camera.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    duration = db.Column(db.Integer)  # in seconds
    file_path = db.Column(db.String(256), nullable=False)
    file_size = db.Column(db.Integer)  # in bytes
    is_archived = db.Column(db.Boolean, default=False)
    recording_type = db.Column(db.String(20), default='continuous')  # continuous, motion, manual
    
    # Relationships
    camera = db.relationship('Camera', back_populates='recordings')
    
    def __repr__(self):
        return f'<Recording {self.id} for Camera {self.camera_id}: {self.start_time}>' 