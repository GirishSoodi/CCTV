import cv2
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
import threading
import os

class FaceRecognizer:
    """
    Handles face detection, recognition, and cross-camera matching.
    Uses OpenCV's face detection and feature extraction.
    """
    
    def __init__(self, recognition_threshold=0.6):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.recognition_threshold = recognition_threshold
        
        # Load OpenCV's pre-trained face detector
        model_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
        if not os.path.exists(model_path):
            # If the file doesn't exist in the utils directory, use OpenCV's default path
            model_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_detector = cv2.CascadeClassifier(model_path)
        
        # For tracking individuals across cameras
        self.face_database = {}  # id -> {encoding, last_seen, cameras}
        self.next_id = 1
        self.db_lock = threading.Lock()
        
        # For tracking matches across cameras
        self.cross_camera_matches = []
        self.last_match_cleanup = time.time()
    
    def process_frame(self, frame, camera_id):
        """
        Process a frame to detect and recognize faces.
        Returns face locations and identity information.
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Create face encodings (simplified - using small image patches as "encodings")
        face_encodings = []
        for (x, y, w, h) in faces:
            # Extract face region and resize to standard size
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (50, 50))  # Standard size
            
            # Flatten the image as a simple "encoding"
            encoding = face_roi.flatten().astype(np.float32)
            # Normalize the encoding
            if np.sum(encoding) != 0:
                encoding = encoding / np.linalg.norm(encoding)
            
            face_encodings.append(encoding)
        
        # Match faces against known faces
        identities = []
        current_time = time.time()
        
        for face_encoding in face_encodings:
            identity = self._identify_face(face_encoding, camera_id, current_time)
            identities.append(identity)
        
        # Clean up old matches periodically
        if current_time - self.last_match_cleanup > 60:  # Clean up every minute
            self._cleanup_old_matches()
            self.last_match_cleanup = current_time
        
        return faces, identities
    
    def _identify_face(self, face_encoding, camera_id, current_time):
        """
        Identify a face by comparing its encoding with known faces.
        If it's a new face, add it to the database.
        """
        with self.db_lock:
            # Check if this face matches any known face
            matches = []
            for face_id, data in self.face_database.items():
                # Compare with the primary encoding using Euclidean distance
                distance = np.linalg.norm(data['encoding'] - face_encoding)
                # Convert distance to confidence (lower distance = higher confidence)
                confidence = 1.0 / (1.0 + distance)
                
                if confidence >= self.recognition_threshold:
                    matches.append((face_id, confidence))
            
            # Sort matches by confidence (highest first)
            matches.sort(key=lambda x: x[1], reverse=True)
            
            if matches:
                # Get the best match
                best_match_id, confidence = matches[0]
                
                # Update the face data
                self.face_database[best_match_id]['last_seen'] = current_time
                
                # Add camera to the list if not already present
                if camera_id not in self.face_database[best_match_id]['cameras']:
                    self.face_database[best_match_id]['cameras'].append(camera_id)
                    
                    # If seen in multiple cameras, add to cross-camera matches
                    if len(self.face_database[best_match_id]['cameras']) > 1:
                        self._add_cross_camera_match(best_match_id)
                
                return {
                    'id': str(best_match_id),
                    'confidence': confidence,
                    'is_new': False
                }
            else:
                # This is a new face
                new_id = self.next_id
                self.next_id += 1
                
                self.face_database[new_id] = {
                    'encoding': face_encoding,
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'cameras': [camera_id]
                }
                
                return {
                    'id': str(new_id),
                    'confidence': 1.0,  # New face, so confidence is 1.0
                    'is_new': True
                }
    
    def _add_cross_camera_match(self, face_id):
        """Add a new cross-camera match."""
        face_data = self.face_database[face_id]
        match_entry = {
            'id': str(face_id),
            'cameras': face_data['cameras'].copy(),
            'first_seen': datetime.fromtimestamp(face_data['first_seen']).strftime('%Y-%m-%d %H:%M:%S'),
            'last_seen': datetime.fromtimestamp(face_data['last_seen']).strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': time.time()
        }
        
        # Add to cross-camera matches
        self.cross_camera_matches.append(match_entry)
    
    def _cleanup_old_matches(self, max_age=3600):
        """Clean up old matches (older than max_age seconds)."""
        current_time = time.time()
        
        # Clean up face database
        with self.db_lock:
            ids_to_remove = []
            for face_id, data in self.face_database.items():
                if current_time - data['last_seen'] > max_age:
                    ids_to_remove.append(face_id)
            
            for face_id in ids_to_remove:
                del self.face_database[face_id]
        
        # Clean up cross-camera matches
        self.cross_camera_matches = [match for match in self.cross_camera_matches
                                    if current_time - match['timestamp'] <= max_age]
    
    def get_cross_camera_matches(self):
        """Get the current cross-camera matches."""
        return self.cross_camera_matches
