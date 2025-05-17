import cv2
import time
import threading
import numpy as np
from datetime import datetime
from .productivity_tracker import ProductivityTracker
from .yolov12_detector import YOLOv12Detector

class VideoProcessor:
    """
    Class to process video streams from CCTV cameras.
    Handles frame extraction, face detection, and streaming.
    """
    
    def __init__(self, video_path, face_recognizer=None, fps=20):
        self.video_path = video_path
        self.face_recognizer = face_recognizer  # Keep for backward compatibility
        self.fps = fps
        self.cap = cv2.VideoCapture(video_path)
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()
        self.camera_id = video_path.split('/')[-1] if '/' in video_path else video_path.split('\\')[-1]
        
        # Initialize YOLOv12 detector for person and desk detection
        self.person_detector = YOLOv12Detector(confidence_threshold=0.5)
        
        # Initialize productivity tracker
        self.productivity_tracker = ProductivityTracker()
        
        # Set up default desk zones based on frame size
        success, frame = self.cap.read()
        if success:
            height, width = frame.shape[:2]
            # Create default desk zones (can be customized later)
            self.setup_default_desk_zones(width, height)
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_frames)
        self.is_running = True
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def _process_frames(self):
        """Process frames from the video in a separate thread."""
        frame_interval = 1.0 / self.fps
        last_time = time.time()
        frame_count = 0
        retry_count = 0
        max_retries = 5
        
        print(f"Starting video processing for {self.video_path}")
        
        # Check if video opened successfully
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            # Try reopening the video
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"Failed to open video after retry: {self.video_path}")
                return
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video dimensions: {width}x{height}, Total frames: {total_frames}")
        
        while self.is_running:
            current_time = time.time()
            if current_time - last_time >= frame_interval:
                success, frame = self.cap.read()
                
                if not success:
                    retry_count += 1
                    print(f"Failed to read frame. Retry {retry_count}/{max_retries}")
                    
                    if retry_count >= max_retries:
                        print("Resetting video to beginning after multiple failed reads")
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        retry_count = 0
                        # Create a blank frame as a fallback
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                        cv2.putText(frame, "Restarting video...", (width//4, height//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        time.sleep(0.1)  # Short delay before retrying
                        continue
                else:
                    retry_count = 0  # Reset retry counter on successful read
                    frame_count += 1
                
                # Process the frame with detection
                try:
                    processed_frame = self.process_frame(frame)
                    
                    # Update the current frame with thread safety
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.processed_frame = processed_frame
                    
                    # Print occasional status
                    if frame_count % 100 == 0:
                        print(f"Processed {frame_count} frames from {self.video_path}")
                        
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    # Use original frame if processing fails
                    with self.frame_lock:
                        self.current_frame = frame.copy()
                        self.processed_frame = frame.copy()
                
                last_time = current_time
            else:
                # Sleep to avoid consuming too much CPU
                time.sleep(0.001)
    
    def setup_default_desk_zones(self, width, height):
        """
        Set up initial desk zones - these will be automatically detected during processing.
        This is just a placeholder until the first frame is processed.
        """
        # We'll let the person detector automatically detect desk zones
        # Just create some initial zones for the productivity tracker
        zone_width = width // 3
        zone_height = height // 3
        
        # Add desk zones with some padding - these will be updated later
        self.productivity_tracker.add_desk_zone(1, width//6, height//6, zone_width, zone_height)
        self.productivity_tracker.add_desk_zone(2, width//2, height//6, zone_width, zone_height)
        self.productivity_tracker.add_desk_zone(3, width//6, height//2, zone_width, zone_height)
        self.productivity_tracker.add_desk_zone(4, width//2, height//2, zone_width, zone_height)
    
    def process_frame(self, frame):
        """
        Process a single frame:
        - Detect persons using YOLOv12
        - Automatically detect desk zones
        - Assign person IDs (person1, person2, etc.)
        - Check station status
        - Track productivity based on desk zones
        - Draw bounding boxes, labels, and status info
        """
        # Process the frame with YOLOv12 detector (this will detect persons and desks)
        output_frame, person_detections, station_status = self.person_detector.process_frame(frame)
        
        # Update productivity tracker with the automatically detected desk zones
        desk_zones = self.person_detector.get_desk_zones_for_productivity_tracker()
        
        # Update productivity tracker with these zones
        for desk_id, (x, y, width, height) in desk_zones.items():
            self.productivity_tracker.add_desk_zone(desk_id, x, y, width, height)
        
        # Update productivity for detected persons
        for detection in person_detections:
            x, y, w, h = detection['box']
            
            # Use the assigned person ID from the YOLOv12 detector
            person_id = f"person_{detection.get('person_id', 'unknown')}"
            
            # Update productivity for this person
            productivity_data = self.productivity_tracker.update_productivity(
                person_id, (x, y, w, h)
            )
            
            # Add productivity label (already added by YOLOv12 detector)
            # The detector already adds person ID and confidence
        
        # Add timestamp and camera ID
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(output_frame, f"Camera: {self.camera_id}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(output_frame, timestamp, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return output_frame
    
    def generate_frames(self):
        """Generate frames for the Flask video stream."""
        frame_count = 0
        error_count = 0
        max_errors = 5
        
        # Create a default frame with message in case of initial loading
        height, width = 480, 640  # Default size
        default_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(default_frame, "Loading video stream...", (width//6, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        while True:
            try:
                # Get the latest processed frame with thread safety
                with self.frame_lock:
                    if self.processed_frame is None:
                        # Use default frame if no processed frame is available yet
                        output_frame = default_frame.copy()
                    else:
                        output_frame = self.processed_frame.copy()
                
                # Ensure the frame is not empty
                if output_frame is None or output_frame.size == 0:
                    error_count += 1
                    if error_count > max_errors:
                        print(f"Too many empty frames ({error_count}). Using default frame.")
                        output_frame = default_frame.copy()
                    continue
                else:
                    error_count = 0  # Reset error count on successful frame
                
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    print("Failed to encode frame as JPEG")
                    continue
                    
                # Yield the frame in the format expected by Flask's Response
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Streamed {frame_count} frames from {self.camera_id}")
                
                # Control the frame rate of the stream
                time.sleep(1/self.fps)
                
            except Exception as e:
                print(f"Error in generate_frames: {e}")
                # Provide a fallback frame in case of error
                error_frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(error_frame, f"Stream Error: {str(e)[:30]}...", (width//6, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                ret, buffer = cv2.imencode('.jpg', error_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                time.sleep(0.5)  # Longer delay on error
    
    def stop(self):
        """Stop the video processing thread and release resources."""
        self.is_running = False
        if self.process_thread.is_alive():
            self.process_thread.join(timeout=1.0)
        self.cap.release()
