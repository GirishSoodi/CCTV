import cv2
import numpy as np
import os
from skimage.morphology import erosion
from .desk_detector import DeskDetector

class PersonDetector:
    """
    Class to detect people in video frames using YOLOv4.
    """
    
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Define color constants
        self.person_color = (220, 155, 58)  # BGR format
        self.vacant_color = (0, 0, 200)     # Red for vacant
        self.taken_color = (0, 200, 0)      # Green for taken
        self.station_color = (0, 100, 0)    # Dark green for station
        
        # Load YOLO model
        self.load_yolo_model()
        
        # Initialize desk detector for automatic desk detection
        self.desk_detector = DeskDetector()
        
        # Station coordinates will be automatically detected
        self.station_coordinates = {}
        
        # Flag to track if desk detection has been performed
        self.desks_detected = False
    
    def load_yolo_model(self):
        """Load YOLOv4 model using OpenCV's DNN module."""
        # Get the directory of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, '..', '..', 'models')
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Paths to YOLO files
        weights_path = os.path.join(models_dir, 'yolov4.weights')
        config_path = os.path.join(models_dir, 'yolov4.cfg')
        coco_names_path = os.path.join(models_dir, 'coco.names')
        
        # Define basic COCO classes
        basic_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                       "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                       "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                       "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                       "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                       "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                       "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
                       "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                       "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                       "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        
        # Create coco.names file if it doesn't exist
        if not os.path.exists(coco_names_path):
            try:
                with open(coco_names_path, 'w') as f:
                    for class_name in basic_classes:
                        f.write(f"{class_name}\n")
                print(f"Created {coco_names_path} with default classes")
            except Exception as e:
                print(f"Error creating coco.names file: {e}")
        
        # Initialize model
        try:
            # First try to use the local model files
            if os.path.exists(weights_path) and os.path.exists(config_path):
                print(f"Loading YOLOv4 model from {weights_path} and {config_path}")
                self.net = cv2.dnn.readNet(weights_path, config_path)
            else:
                # If local files don't exist, try to use OpenCV's built-in files
                print("YOLO model files not found locally, trying OpenCV's built-in files")
                try:
                    self.net = cv2.dnn.readNet(
                        cv2.samples.findFile("yolov4.weights"),
                        cv2.samples.findFile("yolov4.cfg")
                    )
                except Exception as e:
                    print(f"Failed to load OpenCV's built-in YOLOv4: {e}")
                    # Fall back to a simpler model like YOLOv3
                    try:
                        self.net = cv2.dnn.readNet(
                            cv2.samples.findFile("yolov3.weights"),
                            cv2.samples.findFile("yolov3.cfg")
                        )
                        print("Loaded YOLOv3 as fallback")
                    except Exception as e2:
                        print(f"Failed to load YOLOv3 fallback: {e2}")
                        # Create a dummy network as last resort
                        print("WARNING: Using dummy detection. Please run download_models.py")
                        self.net = None
            
            # Set backend and target if net was loaded successfully
            if self.net is not None:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                
                # Get output layer names
                self.layer_names = self.net.getLayerNames()
                self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("WARNING: Using dummy detection. Please run download_models.py")
            self.net = None
        
        # Load COCO class names
        try:
            self.classes = []
            if os.path.exists(coco_names_path):
                with open(coco_names_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
            
            # If classes couldn't be loaded, use the basic ones
            if not self.classes:
                self.classes = basic_classes
                
        except Exception as e:
            print(f"Error loading class names: {e}")
            self.classes = basic_classes
    
    def set_station_coordinates(self, station_id, x1, y1, x2, y2):
        """Set coordinates for a station/desk zone."""
        self.station_coordinates[station_id] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
        
    def detect_desks(self, frame):
        """Detect desk zones in the frame."""
        # Use the desk detector to find desk zones
        desk_zones = self.desk_detector.detect_desks(frame)
        
        # Update station coordinates
        self.station_coordinates = desk_zones
        
        # Mark desks as detected
        self.desks_detected = True
        
        return desk_zones
    
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
    
    def multi_ero(self, im, num=5):
        """Perform multiple erosion on the image."""
        for i in range(num):
            im = erosion(im)
        return im
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame using YOLOv4.
        Returns list of person detections with bounding boxes.
        """
        height, width, _ = frame.shape
        detections = []
        
        # If model couldn't be loaded, use a simple fallback detection
        if self.net is None:
            # Simple fallback: just detect a "person" in the center of the frame
            # This is just a placeholder until proper model is loaded
            center_x = width // 2
            center_y = height // 2
            w = width // 4
            h = height // 2
            x = center_x - w // 2
            y = center_y - h // 2
            
            detection = {
                'box': (x, y, w, h),
                'confidence': 0.5,  # Placeholder confidence
                'class_id': 0       # Person class
            }
            detections.append(detection)
            return detections
        
        try:
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run forward pass
            outs = self.net.forward(self.output_layers)
            
            # Process detections
            class_ids = []
            confidences = []
            boxes = []
            
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filter out weak predictions and non-person detections
                    if confidence > self.confidence_threshold and class_id == 0:  # 0 is person class in COCO
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Apply non-maximum suppression
            if boxes and confidences:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
                
                # Prepare results
                if len(indices) > 0:
                    for i in indices.flatten():
                        box = boxes[i]
                        x, y, w, h = box
                        
                        detection = {
                            'box': (x, y, w, h),
                            'confidence': confidences[i],
                            'class_id': class_ids[i]
                        }
                        detections.append(detection)
        except Exception as e:
            print(f"Error during person detection: {e}")
            # Fallback to a simple detection
            center_x = width // 2
            center_y = height // 2
            w = width // 4
            h = height // 2
            x = center_x - w // 2
            y = center_y - h // 2
            
            detection = {
                'box': (x, y, w, h),
                'confidence': 0.5,  # Placeholder confidence
                'class_id': 0       # Person class
            }
            detections.append(detection)
        
        return detections
    
    def check_station_status(self, detections):
        """
        Check if stations are occupied based on person detections.
        Returns a dictionary with station IDs and their status.
        """
        station_status = {}
        
        # Initialize all stations as vacant
        for station_id in self.station_coordinates:
            station_status[station_id] = 'vacant'
        
        # Check each person detection against station zones
        for detection in detections:
            x, y, w, h = detection['box']
            person_box = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h}
            
            for station_id, coords in self.station_coordinates.items():
                station_box = coords
                
                # Calculate IoU between person and station
                iou = self.get_iou(person_box, station_box)
                
                # If IoU is above threshold, mark station as taken
                if iou > 0.1:  # Threshold can be adjusted
                    station_status[station_id] = 'taken'
        
        return station_status
    
    def process_frame(self, frame):
        """
        Process a frame to detect persons and check station status.
        Returns processed frame with annotations and detection results.
        """
        # Create a copy of the frame to draw on
        output_frame = frame.copy()
        
        # Automatically detect desks if not already done
        if not self.desks_detected or not self.station_coordinates:
            self.detect_desks(frame)
        
        # Detect persons in the frame
        detections = self.detect_persons(frame)
        
        # Check station status
        station_status = self.check_station_status(detections)
        
        # Draw desk zones using the desk detector
        output_frame = self.desk_detector.draw_desk_zones(output_frame, station_status)
        
        # Draw person detections
        for detection in detections:
            x, y, w, h = detection['box']
            confidence = detection['confidence']
            
            # Draw rectangle around the person
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), self.person_color, 2)
            
            # Add confidence label
            label = f"Person: {confidence:.2f}"
            cv2.putText(output_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.person_color, 2)
        
        return output_frame, detections, station_status
