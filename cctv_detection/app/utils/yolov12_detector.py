import os
import cv2
import numpy as np
import torch
from collections import defaultdict
import time

class YOLOv12Detector:
    """
    Class to detect people and desks using YOLOv12.
    Implements person tracking with numeric IDs.
    """
    
    def __init__(self, confidence_threshold=0.5, iou_threshold=0.45):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Define color constants
        self.person_color = (220, 155, 58)  # BGR format
        self.vacant_color = (0, 0, 200)     # Red for vacant
        self.taken_color = (0, 200, 0)      # Green for taken
        self.station_color = (0, 100, 0)    # Dark green for station
        
        # Load YOLOv12 model
        self.model = self.load_yolov12_model()
        
        # Store detected desk zones
        self.desk_zones = {}
        self.desk_detected = False
        
        # Person tracking
        self.person_trackers = {}  # Dictionary to track persons across frames
        self.next_person_id = 1    # Counter for assigning unique IDs
        self.track_history = defaultdict(list)  # Track history for each person
        self.max_track_history = 30  # Maximum number of positions to keep in history
        
        # Dictionary to store person positions for tracking
        self.last_positions = {}
        self.position_history = {}
        
        # Track when persons were last seen
        self.last_seen = {}
        self.max_unseen_frames = 30  # Remove tracker after this many frames of not being seen
    
    def load_yolov12_model(self):
        """Load YOLOv12 model using PyTorch."""
        try:
            # Get the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, '..', '..', 'models')
            
            # Check for YOLOv12 models
            yolov12_dir = os.path.join(models_dir, 'yolov12')
            if os.path.exists(yolov12_dir):
                print(f"Found YOLOv12 directory at {yolov12_dir}")
                
                # First, check for .pt files in the models directory
                pt_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                
                # If no .pt files in models dir, look in the yolov12/weights directory
                if not pt_files:
                    weights_dir = os.path.join(yolov12_dir, 'weights')
                    if os.path.exists(weights_dir):
                        pt_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
                        if pt_files:
                            # Copy the first .pt file to the models directory for easier access
                            import shutil
                            src_path = os.path.join(weights_dir, pt_files[0])
                            dst_path = os.path.join(models_dir, pt_files[0])
                            shutil.copy2(src_path, dst_path)
                            print(f"Copied {pt_files[0]} from weights directory to models directory")
                            pt_files = [pt_files[0]]  # Update pt_files to include only the copied file
                
                if pt_files:
                    model_path = os.path.join(models_dir, pt_files[0])
                    print(f"Loading YOLOv12 model from {model_path}")
                    
                    try:
                        # Try using ultralytics YOLO
                        from ultralytics import YOLO
                        model = YOLO(model_path)
                        print("Successfully loaded model with ultralytics YOLO")
                        return model
                    except Exception as e:
                        print(f"Error loading with ultralytics: {e}")
                        # Fall back to torch hub
                        try:
                            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                            print("Successfully loaded model with torch.hub")
                            return model
                        except Exception as e:
                            print(f"Error loading YOLOv12 model with torch.hub: {e}")
                else:
                    print("No .pt model files found. Downloading a default model...")
            
            # If we get here, we couldn't find or load a YOLOv12 model
            # Try loading a default model from ultralytics
            try:
                from ultralytics import YOLO
                print("Loading default YOLOv8n model from ultralytics")
                model = YOLO("yolov8n.pt")
                print("Successfully loaded default YOLOv8n model")
                return model
            except Exception as e:
                print(f"Error loading default YOLOv8 model: {e}")
                
                # Last resort: try loading from torch hub
                try:
                    print("Attempting to load YOLOv5 from torch hub as last resort")
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                    print("Successfully loaded YOLOv5s model from torch hub")
                    return model
                except Exception as e:
                    print(f"All model loading attempts failed: {e}")
                    return None
                
        except Exception as e:
            print(f"Error in load_yolov12_model: {e}")
            return None
    
    def detect_objects(self, frame):
        """
        Detect objects (people and desks) in a frame using YOLOv12.
        Returns list of detections with bounding boxes.
        """
        if self.model is None:
            print("No model loaded. Attempting to load model again...")
            self.model = self.load_yolov12_model()
            if self.model is None:
                print("Still couldn't load model. Returning empty detections.")
                return [], []
        
        try:
            # Check if frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame received for detection")
                return [], []
                
            # Make a copy of the frame to avoid modifying the original
            frame_copy = frame.copy()
            
            # Convert BGR to RGB for PyTorch model
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            
            # Run inference with confidence threshold
            results = self.model(rgb_frame, conf=self.confidence_threshold)
            
            # Extract detections
            person_detections = []
            desk_detections = []
            
            # Process results based on the model type
            if hasattr(results, 'xyxy'):
                # torch.hub model format (YOLOv5)
                print("Processing YOLOv5 format results")
                predictions = results.xyxy[0]
                
                for pred in predictions:
                    x1, y1, x2, y2, conf, cls = pred
                    
                    # Convert tensor values to Python types
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    conf, cls = float(conf), int(cls)
                    
                    # Get class name
                    cls_name = results.names[cls]
                    
                    # Create detection dictionary
                    detection = {
                        'box': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        'confidence': conf,
                        'class_id': cls,
                        'class_name': cls_name
                    }
                    
                    # Separate person and desk/chair detections
                    if cls_name == 'person':
                        person_detections.append(detection)
                    elif cls_name in ['desk', 'chair', 'dining table', 'bench']:
                        desk_detections.append(detection)
            else:
                # ultralytics YOLO format (YOLOv8/YOLOv12)
                print("Processing ultralytics YOLO format results")
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        try:
                            # Get box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Get confidence and class
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            cls_name = r.names[cls]
                            
                            # Create detection dictionary
                            detection = {
                                'box': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                'confidence': conf,
                                'class_id': cls,
                                'class_name': cls_name
                            }
                            
                            # Separate person and desk/chair/table detections
                            if cls_name == 'person':
                                person_detections.append(detection)
                            elif cls_name in ['desk', 'chair', 'dining table', 'bench']:
                                desk_detections.append(detection)
                        except Exception as box_error:
                            print(f"Error processing box: {box_error}")
                            continue
            
            print(f"Detected {len(person_detections)} persons and {len(desk_detections)} desks/chairs")
            return person_detections, desk_detections
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def assign_person_ids(self, person_detections):
        """
        Assign consistent IDs to detected persons across frames.
        Uses simple position-based tracking.
        """
        current_time = time.time()
        assigned_detections = []
        unassigned_detections = []
        assigned_ids = set()
        
        # First, try to match detections with existing trackers
        for detection in person_detections:
            x, y, w, h = detection['box']
            center_x, center_y = x + w // 2, y + h // 2
            
            best_match_id = None
            best_match_distance = float('inf')
            
            # Find the closest tracker
            for person_id, pos in self.last_positions.items():
                if person_id in assigned_ids:
                    continue
                    
                tracker_x, tracker_y = pos
                distance = np.sqrt((center_x - tracker_x)**2 + (center_y - tracker_y)**2)
                
                # If distance is below threshold and better than previous matches
                if distance < 100 and distance < best_match_distance:
                    best_match_id = person_id
                    best_match_distance = distance
            
            if best_match_id is not None:
                # Update existing tracker
                self.last_positions[best_match_id] = (center_x, center_y)
                self.last_seen[best_match_id] = current_time
                
                # Add position to track history
                self.track_history[best_match_id].append((center_x, center_y))
                if len(self.track_history[best_match_id]) > self.max_track_history:
                    self.track_history[best_match_id].pop(0)
                
                # Add ID to detection
                detection['person_id'] = best_match_id
                assigned_detections.append(detection)
                assigned_ids.add(best_match_id)
            else:
                unassigned_detections.append(detection)
        
        # Create new trackers for unassigned detections
        for detection in unassigned_detections:
            x, y, w, h = detection['box']
            center_x, center_y = x + w // 2, y + h // 2
            
            # Assign new ID
            new_id = self.next_person_id
            self.next_person_id += 1
            
            # Initialize tracker
            self.last_positions[new_id] = (center_x, center_y)
            self.last_seen[new_id] = current_time
            self.track_history[new_id] = [(center_x, center_y)]
            
            # Add ID to detection
            detection['person_id'] = new_id
            assigned_detections.append(detection)
        
        # Remove old trackers that haven't been seen recently
        ids_to_remove = []
        for person_id, last_time in self.last_seen.items():
            if current_time - last_time > 2.0:  # 2 seconds threshold
                ids_to_remove.append(person_id)
        
        for person_id in ids_to_remove:
            self.last_positions.pop(person_id, None)
            self.last_seen.pop(person_id, None)
            self.track_history.pop(person_id, None)
        
        return assigned_detections
    
    def detect_desks(self, frame, desk_detections):
        """
        Detect desk zones in the frame.
        Uses desk detections from YOLOv12 if available, otherwise uses image processing.
        """
        # If we already have desk detections from YOLO, use those
        if desk_detections:
            desk_zones = {}
            for i, detection in enumerate(desk_detections):
                x, y, w, h = detection['box']
                desk_zones[f'station_{i+1}'] = {
                    'x1': x,
                    'y1': y,
                    'x2': x + w,
                    'y2': y + h
                }
            self.desk_zones = desk_zones
            self.desk_detected = True
            return desk_zones
        
        # If we already detected desks before, return those
        if self.desk_detected and self.desk_zones:
            return self.desk_zones
        
        # Otherwise, create default desk zones based on image quadrants
        height, width = frame.shape[:2]
        
        # Create 4 desk zones in a 2x2 grid
        desk_zones = {}
        desk_zones['station_1'] = {'x1': width//4, 'y1': height//4, 'x2': width//2, 'y2': height//2}
        desk_zones['station_2'] = {'x1': width//2, 'y1': height//4, 'x2': 3*width//4, 'y2': height//2}
        desk_zones['station_3'] = {'x1': width//4, 'y1': height//2, 'x2': width//2, 'y2': 3*height//4}
        desk_zones['station_4'] = {'x1': width//2, 'y1': height//2, 'x2': 3*width//4, 'y2': 3*height//4}
        
        self.desk_zones = desk_zones
        self.desk_detected = True
        return desk_zones
    
    def check_station_status(self, person_detections):
        """
        Check if stations are occupied based on person detections.
        Returns a dictionary with station IDs and their status.
        """
        station_status = {}
        
        # Initialize all stations as vacant
        for station_id in self.desk_zones:
            station_status[station_id] = 'vacant'
        
        # Check each person detection against station zones
        for detection in person_detections:
            x, y, w, h = detection['box']
            person_box = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h}
            
            for station_id, coords in self.desk_zones.items():
                station_box = coords
                
                # Calculate IoU between person and station
                iou = self.get_iou(person_box, station_box)
                
                # If IoU is above threshold, mark station as taken
                if iou > 0.1:  # Threshold can be adjusted
                    station_status[station_id] = 'taken'
        
        return station_status
    
    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate area of intersection rectangle
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate area of both bounding boxes
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # Calculate IoU
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou
    
    def draw_desk_zones(self, frame, station_status=None):
        """
        Draw desk zones on the frame with status indicators.
        """
        output_frame = frame.copy()
        
        for station_id, coords in self.desk_zones.items():
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Determine color based on status
            if station_status and station_id in station_status:
                color = self.taken_color if station_status[station_id] == 'taken' else self.vacant_color
            else:
                color = self.station_color
            
            # Draw rectangle for station
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add station ID label
            status_text = f": {station_status[station_id]}" if station_status and station_id in station_status else ""
            cv2.putText(output_frame, f"{station_id}{status_text}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output_frame
    
    def draw_tracks(self, frame, person_detections):
        """Draw person tracks on the frame."""
        output_frame = frame.copy()
        
        # Draw tracks
        for person_id, track in self.track_history.items():
            if len(track) < 2:
                continue
                
            # Draw track line
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output_frame, [points], False, (0, 255, 255), 2)
        
        return output_frame
    
    def process_frame(self, frame):
        """
        Process a frame to detect persons and desks, and check station status.
        Returns processed frame with annotations and detection results.
        """
        # Create a copy of the frame to draw on
        output_frame = frame.copy()
        
        # Detect objects in the frame
        person_detections, desk_detections = self.detect_objects(frame)
        
        # Assign person IDs
        person_detections = self.assign_person_ids(person_detections)
        
        # Detect desk zones
        self.detect_desks(frame, desk_detections)
        
        # Check station status
        station_status = self.check_station_status(person_detections)
        
        # Draw desk zones
        output_frame = self.draw_desk_zones(output_frame, station_status)
        
        # Draw person tracks
        output_frame = self.draw_tracks(output_frame, person_detections)
        
        # Draw person detections with IDs
        for detection in person_detections:
            x, y, w, h = detection['box']
            confidence = detection['confidence']
            person_id = detection.get('person_id', 'unknown')
            
            # Draw rectangle around the person
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), self.person_color, 2)
            
            # Add ID and confidence label
            label = f"Person {person_id}: {confidence:.2f}"
            cv2.putText(output_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.person_color, 2)
        
        return output_frame, person_detections, station_status
    
    def get_desk_zones_for_productivity_tracker(self):
        """
        Convert desk zones to the format expected by the productivity tracker.
        """
        productivity_zones = {}
        
        for station_id, coords in self.desk_zones.items():
            # Extract numeric ID from station_id (e.g., 'station_1' -> 1)
            try:
                desk_id = int(station_id.split('_')[1])
            except (IndexError, ValueError):
                desk_id = len(productivity_zones) + 1
                
            # Convert to (x, y, width, height) format
            x = coords['x1']
            y = coords['y1']
            width = coords['x2'] - coords['x1']
            height = coords['y2'] - coords['y1']
            
            productivity_zones[desk_id] = (x, y, width, height)
        
        return productivity_zones
