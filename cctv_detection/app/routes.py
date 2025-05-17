from flask import Blueprint, render_template, Response, jsonify, request
from .utils.video_processor import VideoProcessor
import os
import cv2
import threading
from .utils.productivity_tracker import ProductivityTracker
from .utils.yolov12_detector import YOLOv12Detector

# Try to import face recognition, but make it optional
try:
    from .utils.face_recognition_utils import FaceRecognizer
    face_recognition_available = True
except ImportError:
    face_recognition_available = False

main = Blueprint('main', __name__)

# Initialize video processors for each camera feed
video_processors = {}

# For backward compatibility, initialize face recognizer if available
if face_recognition_available:
    face_recognizer = FaceRecognizer()
else:
    face_recognizer = None

# Global productivity tracker for cross-camera productivity data
global_productivity_tracker = ProductivityTracker()

# Path to dataset videos
DATASET_PATH = 'd:/Main/Dataset'

# Get list of available video files
def get_available_videos():
    videos = []
    for file in os.listdir(DATASET_PATH):
        if file.endswith('.avi'):
            videos.append(file)
    return videos

@main.route('/')
def index():
    videos = get_available_videos()
    return render_template('index.html', videos=videos, title="CCTV Person Detection with YOLOv12")

@main.route('/productivity')
def productivity_dashboard():
    """Productivity monitoring dashboard with automatic desk detection."""
    videos = get_available_videos()
    return render_template('productivity.html', videos=videos, title="Productivity Monitoring with YOLOv12")

@main.route('/video_feed/<video_id>')
def video_feed(video_id):
    """Video streaming route for a specific camera."""
    if video_id not in video_processors:
        video_path = os.path.join(DATASET_PATH, video_id)
        video_processors[video_id] = VideoProcessor(video_path)
        
    return Response(
        video_processors[video_id].generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@main.route('/start_processing', methods=['POST'])
def start_processing():
    """Start processing selected video feeds."""
    selected_videos = request.json.get('videos', [])
    
    # Clear existing processors
    for processor in video_processors.values():
        processor.stop()
    video_processors.clear()
    
    # Initialize new processors for selected videos
    for video_id in selected_videos:
        video_path = os.path.join(DATASET_PATH, video_id)
        video_processors[video_id] = VideoProcessor(video_path)
    
    return jsonify({'status': 'success', 'message': f'Started processing {len(selected_videos)} video feeds'})

@main.route('/get_matches')
def get_matches():
    """Get current face matches across cameras."""
    # This is kept for backward compatibility
    if face_recognition_available and face_recognizer is not None:
        matches = face_recognizer.get_cross_camera_matches()
        return jsonify(matches)
    else:
        return jsonify([])

@main.route('/get_productivity_data')
def get_productivity_data():
    """Get productivity data for all individuals."""
    productivity_data = []
    
    # Collect productivity data from all video processors
    for video_id, processor in video_processors.items():
        # Get data from this camera's productivity tracker
        camera_data = processor.productivity_tracker.get_productivity_data()
        
        # Add camera ID to each entry
        for entry in camera_data:
            entry['camera_id'] = video_id
            productivity_data.append(entry)
    
    return jsonify(productivity_data)

@main.route('/get_desk_zones')
def get_desk_zones():
    """Get all desk zones from all cameras."""
    desk_zones = {}
    station_zones = {}
    
    for video_id, processor in video_processors.items():
        # Get desk zones from productivity tracker
        desk_zones[video_id] = processor.productivity_tracker.desk_zones
        
        # Get automatically detected station zones from person detector
        station_zones[video_id] = processor.person_detector.station_coordinates
    
    return jsonify({'desk_zones': desk_zones, 'station_zones': station_zones})

@main.route('/update_desk_zone', methods=['POST'])
def update_desk_zone():
    """Update a desk zone for a specific camera."""
    data = request.json
    camera_id = data.get('camera_id')
    desk_id = data.get('desk_id')
    x = data.get('x')
    y = data.get('y')
    width = data.get('width')
    height = data.get('height')
    
    if not all([camera_id, desk_id, x, y, width, height]):
        return jsonify({'status': 'error', 'message': 'Missing required parameters'})
    
    if camera_id not in video_processors:
        return jsonify({'status': 'error', 'message': f'Camera {camera_id} not found'})
    
    # Update the desk zone in productivity tracker
    video_processors[camera_id].productivity_tracker.add_desk_zone(
        desk_id, x, y, width, height
    )
    
    # Also update the station coordinates for person detector
    video_processors[camera_id].person_detector.set_station_coordinates(
        f'station_{desk_id}', x, y, x + width, y + height
    )
    
    return jsonify({'status': 'success', 'message': f'Updated desk zone {desk_id} for camera {camera_id}'})

@main.route('/dashboard')
def dashboard():
    """Dashboard for monitoring detected individuals and station status."""
    videos = get_available_videos()
    return render_template('dashboard.html', videos=videos, title="YOLOv12 Person Detection Dashboard")
