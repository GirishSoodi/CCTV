import cv2
import numpy as np
from skimage.morphology import erosion

class DeskDetector:
    """
    Class to automatically detect desk/station zones in CCTV footage.
    Uses image processing techniques to identify potential desk areas.
    """
    
    def __init__(self):
        # Define color constants
        self.vacant_color = (0, 0, 200)     # Red for vacant
        self.taken_color = (0, 200, 0)      # Green for taken
        self.station_color = (0, 100, 0)    # Dark green for station
        
        # Store detected desk zones
        self.desk_zones = {}
        
        # Detection parameters
        self.min_desk_area = 5000  # Minimum area for a desk zone
        self.max_desk_area = 100000  # Maximum area for a desk zone
        
    def multi_ero(self, im, num=5):
        """Perform multiple erosion on the image."""
        result = im.copy()
        for i in range(num):
            result = erosion(result)
        return result
    
    def detect_desks(self, frame):
        """
        Detect potential desk zones in a frame.
        Uses edge detection and contour analysis.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and shape
        desk_zones = {}
        desk_id = 1
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_desk_area < area < self.max_desk_area:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if rectangle has desk-like proportions
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:  # Typical desk aspect ratios
                    desk_zones[f'station_{desk_id}'] = {
                        'x1': x,
                        'y1': y,
                        'x2': x + w,
                        'y2': y + h
                    }
                    desk_id += 1
        
        # If no desks detected, create some default zones based on image quadrants
        if not desk_zones:
            height, width = frame.shape[:2]
            
            # Create 4 desk zones in a 2x2 grid
            desk_zones['station_1'] = {'x1': width//4, 'y1': height//4, 'x2': width//2, 'y2': height//2}
            desk_zones['station_2'] = {'x1': width//2, 'y1': height//4, 'x2': 3*width//4, 'y2': height//2}
            desk_zones['station_3'] = {'x1': width//4, 'y1': height//2, 'x2': width//2, 'y2': 3*height//4}
            desk_zones['station_4'] = {'x1': width//2, 'y1': height//2, 'x2': 3*width//4, 'y2': 3*height//4}
        
        # Update the stored desk zones
        self.desk_zones = desk_zones
        
        return desk_zones
    
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
