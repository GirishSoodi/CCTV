# Real-Time Person Detection from CCTV Feeds

A Flask-based application for real-time person detection, station monitoring, and productivity tracking from CCTV feeds, using the Edinburgh office monitoring video dataset.

## Features

- Real-time streaming and processing of CCTV footage
- Person detection using YOLOv4 object detection
- Station/desk zone occupancy monitoring
- Productivity tracking based on person location
- Interactive dashboard for monitoring detections and station status
- Support for multiple simultaneous video feeds

## Project Structure

```
cctv_detection/
├── app/
│   ├── __init__.py          # Flask application initialization
│   ├── routes.py            # Application routes and endpoints
│   ├── static/              # Static assets (CSS, JS)
│   │   ├── css/
│   │   │   └── style.css    # Custom styles
│   │   └── js/
│   │       └── main.js      # JavaScript functionality
│   ├── templates/           # HTML templates
│   │   ├── base.html        # Base template with common elements
│   │   ├── index.html       # Main page with video feeds
│   │   ├── productivity.html # Productivity monitoring page
│   │   └── dashboard.html   # Analytics dashboard
│   └── utils/               # Utility modules
│       ├── face_recognition_utils.py  # Legacy face detection (backward compatibility)
│       ├── person_detector.py         # YOLOv4-based person detection
│       ├── productivity_tracker.py    # Productivity and desk zone tracking
│       └── video_processor.py         # Video stream processing
├── models/                  # YOLOv4 model files
│   ├── yolov4.cfg           # YOLOv4 configuration
│   ├── yolov4.weights       # YOLOv4 pre-trained weights
│   └── coco.names           # COCO dataset class names
├── requirements.txt         # Python dependencies
├── download_models.py       # Script to download YOLOv4 models
└── run.py                   # Application entry point
```

## Requirements

- Python 3.7+
- Flask
- OpenCV
- NumPy
- scikit-image
- matplotlib
- tqdm
- Flask-SocketIO

*Note: face_recognition library is optional and only needed for backward compatibility*

## Installation

1. Clone the repository or download the source code
2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download the YOLOv4 model files (recommended but optional):
   ```
   python download_models.py
   ```
   *Note: If you skip this step, the system will use a fallback detection method*

## Usage

1. Ensure your Edinburgh office monitoring video dataset is in the `d:\Main\Dataset` directory
2. Run the application:
   ```
   python run.py
   ```
3. Open your web browser and navigate to `http://localhost:5000`
4. Select the camera feeds you want to monitor and click "Start Processing"
5. View the dashboard at `http://localhost:5000/dashboard` for analytics and station status
6. View the productivity tracker at `http://localhost:5000/productivity` for monitoring desk zone occupancy

## System Components

### Video Processing
- Streams video frames from AVI files
- Processes frames in real-time for person detection
- Supports multiple simultaneous video feeds

### Person Detection
- Uses YOLOv4 for accurate person detection
- Identifies station/desk occupancy status
- Color-coded visualization of persons and stations

### Station Monitoring
- Tracks station/desk occupancy in real-time
- Displays status as vacant or taken
- Calculates occupancy metrics for each station

### Productivity Tracking
- Monitors person presence in designated desk zones
- Calculates productivity scores based on zone presence
- Provides productivity analytics per individual

### Dashboard
- Real-time metrics on detected persons
- Station occupancy visualization
- Productivity analytics and zone monitoring

## Evaluation Metrics

- Person Detection Accuracy: Precision, Recall, and F1-Score
- Station Occupancy Accuracy: Correctly identifying vacant vs. taken stations
- Latency: Time taken to process frames and detect persons
- Productivity Tracking Accuracy: Correctly associating persons with desk zones
- Scalability Performance: Handling multiple streams concurrently
