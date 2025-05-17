# import os
# import urllib.request
# import sys
# import zipfile
# import shutil
# import subprocess

# def download_file(url, destination):
#     """Download a file from a URL to a destination path."""
#     print(f"Downloading {url} to {destination}...")
    
#     # Create directory if it doesn't exist
#     os.makedirs(os.path.dirname(destination), exist_ok=True)
    
#     # Download the file
#     try:
#         urllib.request.urlretrieve(url, destination)
#         print(f"Successfully downloaded to {destination}")
#         return True
#     except Exception as e:
#         print(f"Error downloading {url}: {e}")
#         return False

# def download_github_repo(repo_url, branch="main"):
#     """Download a GitHub repository as a zip file and extract it."""
#     # Create temp directory for downloads
#     temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
#     os.makedirs(temp_dir, exist_ok=True)
    
#     # Format the download URL for the zip file
#     zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
#     zip_path = os.path.join(temp_dir, f"{branch}.zip")
    
#     print(f"Downloading repository from {zip_url}...")
    
#     try:
#         # Download the zip file
#         urllib.request.urlretrieve(zip_url, zip_path)
        
#         # Extract the zip file
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(temp_dir)
        
#         # Get the name of the extracted directory
#         extracted_dir = None
#         for item in os.listdir(temp_dir):
#             item_path = os.path.join(temp_dir, item)
#             if os.path.isdir(item_path) and item.endswith(f"-{branch}"):
#                 extracted_dir = item_path
#                 break
        
#         if not extracted_dir:
#             print("Error: Could not find extracted directory")
#             return None
        
#         print(f"Successfully downloaded and extracted repository to {extracted_dir}")
#         return extracted_dir
        
#     except Exception as e:
#         print(f"Error downloading repository: {e}")
#         return None

# def main():
#     # Create models directory
#     models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
#     os.makedirs(models_dir, exist_ok=True)
    
#     print("\n=== Downloading YOLOv12 from GitHub ===\n")
    
#     # Download YOLOv12 repository
#     repo_url = "https://github.com/sunsmarterjie/yolov12"
#     extracted_dir = download_github_repo(repo_url, branch="main")
    
#     if not extracted_dir:
#         print("Failed to download YOLOv12 repository. Falling back to YOLOv4...")
        
#         # URLs for YOLOv4 files as fallback
#         yolov4_cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"
#         yolov4_weights_url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"
#         coco_names_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
        
#         # Destination paths
#         yolov4_cfg_path = os.path.join(models_dir, "yolov4.cfg")
#         yolov4_weights_path = os.path.join(models_dir, "yolov4.weights")
#         coco_names_path = os.path.join(models_dir, "coco.names")
        
#         # Download files
#         success = True
        
#         if not os.path.exists(yolov4_cfg_path):
#             success &= download_file(yolov4_cfg_url, yolov4_cfg_path)
#         else:
#             print(f"{yolov4_cfg_path} already exists, skipping download.")
        
#         if not os.path.exists(coco_names_path):
#             success &= download_file(coco_names_url, coco_names_path)
#         else:
#             print(f"{coco_names_path} already exists, skipping download.")
        
#         if not os.path.exists(yolov4_weights_path):
#             print("Downloading YOLOv4 weights (this may take a while)...")
#             success &= download_file(yolov4_weights_url, yolov4_weights_path)
#         else:
#             print(f"{yolov4_weights_path} already exists, skipping download.")
#     else:
#         # Copy YOLOv12 model files to our models directory
#         print("\nCopying YOLOv12 model files to models directory...")
        
#         # Find the YOLOv12 model files
#         yolov12_dir = os.path.join(extracted_dir, 'weights')
        
#         if os.path.exists(yolov12_dir):
#             # Look for .pt files (PyTorch models)
#             pt_files = [f for f in os.listdir(yolov12_dir) if f.endswith('.pt')]
            
#             if pt_files:
#                 # Copy the model files to our models directory
#                 for pt_file in pt_files:
#                     src_path = os.path.join(yolov12_dir, pt_file)
#                     dst_path = os.path.join(models_dir, pt_file)
#                     shutil.copy2(src_path, dst_path)
#                     print(f"Copied {pt_file} to {dst_path}")
                
#                 # Copy the class names file if it exists
#                 coco_names_path = os.path.join(extracted_dir, 'data', 'coco.names')
#                 if os.path.exists(coco_names_path):
#                     dst_path = os.path.join(models_dir, 'coco.names')
#                     shutil.copy2(coco_names_path, dst_path)
#                     print(f"Copied coco.names to {dst_path}")
#                 else:
#                     # Create a basic coco.names file
#                     create_coco_names(os.path.join(models_dir, 'coco.names'))
                
#                 print("\nYOLOv12 model files copied successfully!")
                
#                 # Copy the entire repository for reference
#                 yolov12_models_dir = os.path.join(models_dir, 'yolov12')
#                 if os.path.exists(yolov12_models_dir):
#                     shutil.rmtree(yolov12_models_dir)
#                 shutil.copytree(extracted_dir, yolov12_models_dir)
#                 print(f"Copied YOLOv12 repository to {yolov12_models_dir}")
#             else:
#                 print("No .pt model files found in the YOLOv12 repository.")
#                 print("Please download the model weights manually from the repository releases.")
#         else:
#             print(f"Could not find weights directory in {extracted_dir}")
#             print("Please download the model weights manually from the repository releases.")
    
#     # Clean up temp directory
#     temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
#     if os.path.exists(temp_dir):
#         try:
#             shutil.rmtree(temp_dir)
#             print("Cleaned up temporary files.")
#         except Exception as e:
#             print(f"Error cleaning up temporary files: {e}")
    
#     print("\nModel download process completed!")

# def create_coco_names(path):
#     """Create a basic coco.names file with common object classes."""
#     classes = [
#         "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#         "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#         "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
#         "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
#         "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
#         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
#         "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
#         "sofa", "pottedplant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
#         "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
#         "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "desk"
#     ]
    
#     with open(path, 'w') as f:
#         for class_name in classes:
#             f.write(f"{class_name}\n")

# if __name__ == "__main__":
#     main()


import os
import urllib.request
import zipfile
import shutil

YOLOV12_REPO = "https://github.com/sunsmarterjie/yolov12"
YOLOV12_BRANCH = "main"

def download_file(url, destination):
    print(f"Downloading {url} to {destination} ...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded: {destination}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def download_github_repo_zip(repo_url, branch, temp_dir):
    zip_url = f"{repo_url}/archive/refs/heads/{branch}.zip"
    zip_path = os.path.join(temp_dir, f"{branch}.zip")
    print(f"Fetching repository zip from {zip_url}")
    try:
        urllib.request.urlretrieve(zip_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        extracted_dir = os.path.join(temp_dir, f"yolov12-{branch}")
        if os.path.isdir(extracted_dir):
            print(f"Extracted to: {extracted_dir}")
            return extracted_dir
        else:
            print("Extraction failed: Directory not found.")
    except Exception as e:
        print(f"Error fetching repo: {e}")
    return None

def copy_model_files(extracted_dir, models_dir):
    # Find weights directory (.pt files)
    weights_dir = os.path.join(extracted_dir, "weights")
    found = False
    if os.path.isdir(weights_dir):
        pt_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
        for pt_file in pt_files:
            src = os.path.join(weights_dir, pt_file)
            dst = os.path.join(models_dir, pt_file)
            shutil.copy2(src, dst)
            print(f"Copied: {pt_file} to {dst}")
            found = True
    if not found:
        print("No .pt weights found in repo. Please download weights from the [YOLOv12 releases page](https://github.com/sunsmarterjie/yolov12/releases) and place them in the models directory.")
    # Copy coco.names if present
    coco_names_src = os.path.join(extracted_dir, "data", "coco.names")
    if os.path.isfile(coco_names_src):
        shutil.copy2(coco_names_src, os.path.join(models_dir, "coco.names"))
        print("Copied coco.names.")
    else:
        print("coco.names not found. Creating a default one.")
        create_coco_names(os.path.join(models_dir, "coco.names"))

def create_coco_names(path):
    classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "sofa", "pottedplant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "desk"
    ]
    with open(path, 'w') as f:
        for c in classes:
            f.write(f"{c}\n")
    print(f"Created default coco.names at {path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(base_dir, "temp")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    print("\n=== Downloading YOLOv12 from GitHub ===\n")
    extracted_dir = download_github_repo_zip(YOLOV12_REPO, YOLOV12_BRANCH, temp_dir)
    if extracted_dir:
        copy_model_files(extracted_dir, models_dir)
        # Optionally, copy the whole repo for reference
        yolov12_repo_dst = os.path.join(models_dir, "yolov12")
        if os.path.exists(yolov12_repo_dst):
            shutil.rmtree(yolov12_repo_dst)
        shutil.copytree(extracted_dir, yolov12_repo_dst)
        print(f"Copied full repo to {yolov12_repo_dst}")
    else:
        print("Failed to download YOLOv12 repository. Please check your internet connection or try again later.")

    # Clean up with retry mechanism
    if os.path.exists(temp_dir):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Try to remove the temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                if not os.path.exists(temp_dir):
                    print("Cleaned up temporary files.")
                    break
                else:
                    print(f"Retry {retry_count+1}/{max_retries} to clean up temporary files...")
                    retry_count += 1
                    # Wait a bit before retrying
                    import time
                    time.sleep(2)
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")
                print("You may need to manually delete the temp directory later.")
                break
        
        if os.path.exists(temp_dir) and retry_count >= max_retries:
            print("Could not clean up all temporary files after multiple attempts.")
            print(f"You may need to manually delete the directory: {temp_dir}")
    print("\nModel download process completed!")

if __name__ == "__main__":
    main()
