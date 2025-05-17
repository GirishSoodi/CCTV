import cv2
import numpy as np
import time
from datetime import datetime
import threading

class ProductivityTracker:
    """
    Tracks productivity based on whether individuals remain in their designated desk zones.
    """
    
    def __init__(self):
        # Dictionary to store desk zones: {desk_id: (x, y, w, h)}
        self.desk_zones = {}
        
        # Dictionary to store productivity scores: {person_id: {score, time_in_zone, last_updated}}
        self.productivity_scores = {}
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Constants for productivity calculation
        self.MAX_SCORE = 100
        self.SCORE_DECAY_RATE = 5  # Points lost per second when outside desk zone
        self.SCORE_GAIN_RATE = 1   # Points gained per second when inside desk zone
    
    def add_desk_zone(self, desk_id, x, y, width, height):
        """Add or update a desk zone."""
        with self.lock:
            self.desk_zones[desk_id] = (x, y, width, height)
    
    def remove_desk_zone(self, desk_id):
        """Remove a desk zone."""
        with self.lock:
            if desk_id in self.desk_zones:
                del self.desk_zones[desk_id]
    
    def is_in_desk_zone(self, person_location, desk_id=None):
        """
        Check if a person is within a specific desk zone or any desk zone.
        person_location: (x, y, w, h) - bounding box of the person
        desk_id: specific desk to check, or None to check all desks
        Returns: (is_in_zone, desk_id) or (False, None) if not in any zone
        """
        px, py, pw, ph = person_location
        # Calculate person center point
        person_center_x = px + pw // 2
        person_center_y = py + ph // 2
        
        with self.lock:
            if desk_id is not None:
                # Check specific desk
                if desk_id in self.desk_zones:
                    dx, dy, dw, dh = self.desk_zones[desk_id]
                    if (dx <= person_center_x <= dx + dw and 
                        dy <= person_center_y <= dy + dh):
                        return True, desk_id
                return False, None
            else:
                # Check all desks
                for d_id, (dx, dy, dw, dh) in self.desk_zones.items():
                    if (dx <= person_center_x <= dx + dw and 
                        dy <= person_center_y <= dy + dh):
                        return True, d_id
                return False, None
    
    def update_productivity(self, person_id, person_location):
        """
        Update productivity score based on whether the person is in their desk zone.
        Returns the updated productivity score.
        """
        current_time = time.time()
        
        # Check if person is in any desk zone
        is_in_zone, desk_id = self.is_in_desk_zone(person_location)
        
        with self.lock:
            # Initialize if this is a new person
            if person_id not in self.productivity_scores:
                self.productivity_scores[person_id] = {
                    'score': self.MAX_SCORE / 2,  # Start at 50%
                    'time_in_zone': 0,
                    'time_out_zone': 0,
                    'last_updated': current_time,
                    'assigned_desk': desk_id if is_in_zone else None
                }
            
            person_data = self.productivity_scores[person_id]
            
            # If person doesn't have an assigned desk yet and is in a zone, assign it
            if person_data['assigned_desk'] is None and is_in_zone:
                person_data['assigned_desk'] = desk_id
            
            # Calculate time elapsed since last update
            time_elapsed = current_time - person_data['last_updated']
            
            # Update productivity score based on zone presence
            if is_in_zone and desk_id == person_data['assigned_desk']:
                # Person is in their assigned desk zone - increase score
                person_data['score'] = min(
                    self.MAX_SCORE, 
                    person_data['score'] + (self.SCORE_GAIN_RATE * time_elapsed)
                )
                person_data['time_in_zone'] += time_elapsed
            else:
                # Person is outside their desk zone - decrease score
                person_data['score'] = max(
                    0, 
                    person_data['score'] - (self.SCORE_DECAY_RATE * time_elapsed)
                )
                person_data['time_out_zone'] += time_elapsed
            
            # Update last updated timestamp
            person_data['last_updated'] = current_time
            
            return {
                'person_id': person_id,
                'score': round(person_data['score']),
                'in_zone': is_in_zone,
                'desk_id': person_data['assigned_desk'],
                'time_in_zone': round(person_data['time_in_zone']),
                'time_out_zone': round(person_data['time_out_zone'])
            }
    
    def get_productivity_data(self, person_id=None):
        """Get productivity data for a specific person or all people."""
        with self.lock:
            if person_id is not None:
                if person_id in self.productivity_scores:
                    data = self.productivity_scores[person_id]
                    return {
                        'person_id': person_id,
                        'score': round(data['score']),
                        'in_zone': self.is_in_desk_zone(person_id)[0],
                        'desk_id': data['assigned_desk'],
                        'time_in_zone': round(data['time_in_zone']),
                        'time_out_zone': round(data['time_out_zone'])
                    }
                return None
            else:
                # Return data for all people
                result = []
                for pid, data in self.productivity_scores.items():
                    result.append({
                        'person_id': pid,
                        'score': round(data['score']),
                        'desk_id': data['assigned_desk'],
                        'time_in_zone': round(data['time_in_zone']),
                        'time_out_zone': round(data['time_out_zone'])
                    })
                return result
    
    def draw_desk_zones(self, frame):
        """Draw desk zones on the frame."""
        with self.lock:
            for desk_id, (x, y, w, h) in self.desk_zones.items():
                # Draw rectangle for desk zone
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Add desk ID label
                cv2.putText(frame, f"Desk {desk_id}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
