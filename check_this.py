# python standard libraries
import os
import time
import glob
import re
import shutil
import pickle
from datetime import datetime, timedelta
from base64 import b64decode, b64encode

# google colab/notebook libraries
from IPython.display import display, Javascript, Image
from IPython.display import Video
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow

# external libraries
import cv2
import numpy as np
import PIL
import io
import html
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import erosion

# define color constants
person_color = (220, 155, 58)
vacant_color = (0, 0, 200)
taken_color = (0, 200, 0)
station_color = (0, 100, 0)

# constant coordinates
coordinates = {
    'station_1' : {'x1':1600, 'x2':1800, 
                   'y1':575, 'y2':780},
    'station_2' : {'x1':1287, 'x2':1472, 
                   'y1':310, 'y2':425},               
    'station_3' : {'x1':1145, 'x2':1287, 
                   'y1':197, 'y2':268},
               
    'station_4' : {'x1':561, 'x2':764, 
                   'y1':424, 'y2':578}
}

coordinates_ocr= [(1256, 39), (1885, 101)]

%matplotlib inline




from IPython.display import HTML
from IPython.display import Image

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<style>
.output_png {
    display: table-cell;
    text-align: center;
    horizontal-align: middle;
    vertical-align: middle;
    margin:auto;
}

tbody, thead {
    margin-left:100px;
}

</style>
<form action="javascript:code_toggle()"><input type="submit"
value="Click here to toggle on/off the raw code."></form>''')


# copy whole dataset
!cp /content/drive/MyDrive/MSDS/ML3/final_project/XVR_ch5_main*.mp4 /content/

# clone of darknet github, extracted dataset, configurations, and custom weights
!cp /content/drive/MyDrive/MSDS/ML3/final_project/darknet_best.zip /content/

# trained weights from custom dataset
!cp /content/drive/MyDrive/MSDS/ML3/final_project/yolov4-obj_best.weights /content/

# unzip darknet directory 
!unzip -qq darknet_best.zip

# clone original darknet repo (use if you want default settings)
!git clone https://github.com/AlexeyAB/darknet

# change makefile to have GPU, OPENCV and LIBSO enabled
%cd /content/darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile

# make darknet (builds darknet so that you can then use the darknet.py file 
# and have its dependencies)
!make


# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

    # get image ratios to convert bounding boxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width/width
    height_ratio = img_height/height

    # run model on darknet style image to get detections
    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image)
    free_image(darknet_image)
    return detections, width_ratio, height_ratio



# extract data from video to 700 sampled images
cap = cv2.VideoCapture('XVR_ch5_main_20220214100004_20220214110005.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
img_array =[]
counter = 0
image_count = 0
while ret:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if counter % int(length/700) == 0:
        fname = f'{image_count}.jpg'
        image_count += 1
        cv2.imwrite(fname, frame)
        
    counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# copy txt and jpg data to a directory
fnames = glob.glob('data/*.txt')

for fname in fnames:
    fname_target = fname.replace('data', 'train')
    shutil.copyfile(fname, fname_target)
    if 'classes' in fname:
        continue
    else:
        image_fname = fname.replace('txt', 'jpg')
        image_target = image_fname.replace('data', 'train')
        shutil.copyfile(image_fname, image_target)
        
# create train and test data
# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open('./data/train.txt', 'w')
file_test = open('./data/test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(os.getcwd(), "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write("data/obj" + "/" + title + '.jpg' + "\n")
    else:
        file_train.write("data/obj" + "/" + title + '.jpg' + "\n")
        counter = counter + 1

file_train.close()
file_test.close()


def non_max_suppression_fast(detections, overlap_thresh):
    """ modified non max suppression from darknet to get the overlap
        with max confidence
    
    Parameters
    ==========
    detections       :     tuple
                           class_name, confidence, and coordinates
    overlap_thresh   :     float
                           IOU threshold
    
    Returns
    ==========
    non_max_suppression_fast   :   tuple
                                   detections without high overlap
    """
    boxes = []
    confs = []

    for detection in detections:
        class_name, conf, (x, y, w, h) = detection
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        boxes.append(np.array([x1, y1, x2, y2]))
        confs.append(conf)
   
    boxes_array = np.array(boxes)

    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes_array[:, 0]
    y1 = boxes_array[:, 1]
    x2 = boxes_array[:, 2]
    y2 = boxes_array[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    confs = np.array(confs)
    # keep looping while some indexes still remain in the indexes
    # list

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # choose the highest confidence among overlaps
        overlap_args = np.where(overlap > overlap_thresh)[0]
        overlap_indices = idxs[overlap_args].tolist() + [i]
        confidence_list = confs[idxs[overlap_args]].tolist() + [confs[i]]
        confidence_list = list(map(float, confidence_list))
        highest_confidence = np.argmax(confidence_list)
        pick.append(overlap_indices[highest_confidence])

        # delete indices that overlaps
        idxs = np.delete(idxs, np.concatenate(([last], overlap_args)))

    return [detections[i] for i in pick]



# import darknet functions to perform object detections
from darknet import *
# load in our YOLOv4 architecture network
(network, 
 class_names, 
 class_colors) = load_network("cfg/yolov4-obj.cfg", 
                              "data/obj.data", 
                              "backup/yolov4-obj_best.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
    """ darknet helper function to get detections, width and height ratio

    Parameters
    ==========
    img           :    np.array
                       image file
    width         :    int
                       width
    height        :    int
                       height
    
    Returns
    =========
    darknet_helper  : tuple
                      tuple of detections, width and height ratio
    """
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

    # get image ratios to convert bounding boxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width/width
    height_ratio = img_height/height

    # run model on darknet style image to get detections
    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image)
    free_image(darknet_image)
    return detections, width_ratio, height_ratio


# run custom yolo on a sample image 
vidcap = cv2.VideoCapture('../XVR_ch5_main_20220214100004_20220214110005.mp4')

for i in range(15):
    success,frame = vidcap.read()

# get the predicted detections of the trained custom yolo
detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

# apply non max suppression to eliminate multiple predictions
# on same person
detections = non_max_suppression_fast(detections, 0.65)

for label, confidence, bbox in detections:
    left, top, right, bottom = bbox2points(bbox)
    left, top, right, bottom = (int(left * width_ratio), int(top * height_ratio), 
    int(right * width_ratio), int(bottom * height_ratio))
    cv2.rectangle(frame, (left, top), (right, bottom), person_color, 2)
    cv2.putText(frame, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    person_color, 2)

cv2_imshow(frame)



# run custom yolo on a sample image 
vidcap = cv2.VideoCapture('/content/XVR_ch5_main_20220214100004_20220214110005.mp4')
success,frame = vidcap.read()


for stations, coordinate in coordinates.items():
    cv2.rectangle(frame, (coordinate['x1'], coordinate['y1']), 
                    (coordinate['x2'], coordinate['y2']), station_color, 2)
    cv2.putText(frame, f"{stations}",
                (coordinate['x1'], coordinate['y1'] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                station_color, 2)

cv2_imshow(frame)

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    
    return iou 


# run custom yolo on a sample image 
vidcap = cv2.VideoCapture('/content/XVR_ch5_main_20220214100004_20220214110005.mp4')
success,frame = vidcap.read()

# get the predicted detections of the trained custom yolo
detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

# apply non max suppression to eliminate multiple predictions
# on same person
detections = non_max_suppression_fast(detections, 0.65)
detections_bb = []
for label, confidence, bbox in detections:
    left, top, right, bottom = bbox2points(bbox)
    left, top, right, bottom = (int(left * width_ratio), 
                                int(top * height_ratio), 
                                int(right * width_ratio), 
                                int(bottom * height_ratio))
    
    cv2.rectangle(frame, (left, top), (right, bottom), person_color, 2)
    cv2.putText(frame, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    person_color, 2)

    detections_bb.append({
        'x1' : left,
        'y1' : top,
        'x2' : right,
        'y2' : bottom
    })

thresh = 0.3
for stations, coordinate in coordinates.items():
    taken = False
    for detection in detections_bb:
        iou = get_iou(coordinate, detection)
        if iou >= thresh:
            taken = True
            break
    color = taken_color if taken else vacant_color
        
    cv2.rectangle(frame, (coordinate['x1'], coordinate['y1']), 
                    (coordinate['x2'], coordinate['y2']), color, 2)
    
    cv2.putText(frame, f"{stations}",
                (coordinate['x1'], coordinate['y1'] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 2)
frame = cv2.resize(frame, (1080, 720), 
                    interpolation=cv2.INTER_AREA)

cv2_imshow(frame)

def multi_ero(im, num):
    """ Perform multiple erosion on the image

    Parameters
    ==========
    im        :     np.array
                    image file
    num       :     int
                    number of times to apply erosion
    """
    for i in range(num):
        im = erosion(im)
    return im

imgs = []
# get images for testing 
vidcap = cv2.VideoCapture('../XVR_ch5_main_20220214100004_20220214110005.mp4')
success,frame = vidcap.read()
for i in tqdm(range(8000)):
    # Capture frame-by-frame
    success, frame = vidcap.read()
    if not i % 50: 
        if frame is not None:
            imgs.append(frame)
        else:
            pass
invalid = []
valid = []
datetime_clean = []


for img in tqdm(imgs):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    contrast = 3
    contrast = max(contrast, 1.0); contrast = min(contrast, 3.0)

    brightness = 60
    brightness = max(brightness, 0.0); brightness = min(brightness, 100.0)

    img = np.clip(contrast * img.astype('float32') 
                        + brightness, 0.0, 255.0)

    img = img.astype('uint8')

    img = cv2.adaptiveThreshold(img,
                                255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY,
                                21, 2)

    img = img[coordinates_ocr[0][1]:coordinates_ocr[1][1], 
              coordinates_ocr[0][0]:coordinates_ocr[1][0]]

    img = multi_ero(img, 2)
    datetime_clean.append(img)
    text = pytesseract.image_to_string(img, lang='eng', 
            config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789:-')
    
    time_format = r'[0-5]\d:[0-5]\d:[0-5]\d'
    date_format = r'\d{4}-(?:0\d|1[12])-(?:[0-2]\d|3[01])'
    datetime_format = date_format + time_format
    text = text.replace(' ', '')
    try:
        timestamp_string = re.sub('(\d{4}-(?:0\d|1[12])-(?:[0-2]\d|3[01]))', 
                                r'\1' + r' ', 
                                re.findall(datetime_format, text)[0])
    except:
        invalid.append(text)
        continue

    
    if len(text) != 20:
        invalid.append(text)
        
    else:
        valid.append(text)




def get_ocr_datetime(img, contrast=3, brightness=60):
    """ get the datetime equivalent based on the image

    Parameters
    ==========
    img        :    np.array
                    image file
    contrast   :    int
                    contrast between 1-3
    brightness :    int 
                    brightness between 0-100

    Returns
    =========
    get_ocr_datetime  :   datetime.datetime
                          datetime equivalent of the cctv image
    """
    # convert to grayscale
    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    contrast = max(contrast, 1.0)
    contrast = min(contrast, 3.0)
    
    brightness = max(brightness, 0.0) 
    brightness = min(brightness, 100.0)

    # clip image based on contrast and brightness provided
    img = np.clip(contrast * img.astype('float32') 
                        + brightness, 0.0, 255.0)

    img = img.astype('uint8')

    # perform adaptive thresholding
    img = cv2.adaptiveThreshold(img,
                              255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv2.THRESH_BINARY,
                              21, 2)

    # perform segmentation on the region of interest
    img = img[coordinates_ocr[0][1]:coordinates_ocr[1][1], 
              coordinates_ocr[0][0]:coordinates_ocr[1][0]]

    # perform multiple erosion
    img = multi_ero(img, 2)
    
    # get text using pytesseract 
    text = pytesseract.image_to_string(img, lang='eng', 
            config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789:-')
    
    # check validity of results
    time_format = r'[0-5]\d:[0-5]\d:[0-5]\d'
    date_format = r'\d{4}-(?:0\d|1[12])-(?:[0-2]\d|3[01])'
    datetime_format = date_format + time_format
    text = text.replace(' ', '')

    if len(text) == 20:
        text = '2022-02-14' + text[10:]
    
    try:
        timestamp_string = re.sub('(\d{4}-(?:0\d|1[12])-(?:[0-2]\d|3[01]))', 
                                r'\1' + r' ', 
                                re.findall(datetime_format, text)[0])
    except:
        return None
    
    return datetime.strptime(timestamp_string, "%Y-%m-%d %H:%M:%S")


# initialize timer per station
# list definition:
# list[0] : total time in work station
# list[1] : last datetime 
# list[2] : debt
timer = {'station_' + str(i): [timedelta(0), None, False] for i in range(1,5)}

%cd /content/
cap = cv2.VideoCapture('XVR_ch5_main_20220214100004_20220214110005.mp4')
success,frame = cap.read()

width =  1600
height = 900
resize = True
img_array =[]
for i in tqdm(range(4500)):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if i <= 2600:
        continue

    detections, width_ratio, height_ratio = darknet_helper(frame, 
                                                           width, 
                                                           height)
    detections = non_max_suppression_fast(detections, 0.65)
    detections_bb = []
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = (int(left * width_ratio), 
                                    int(top * height_ratio), 
                                    int(right * width_ratio), 
                                    int(bottom * height_ratio))
        cv2.rectangle(frame, (left, top), (right, bottom), person_color, 2)
        cv2.putText(frame, "{} [{:.2f}]".format(label, float(confidence)),
                            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            person_color, 4)
        
        detections_bb.append({
                'x1' : left,
                'y1' : top,
                'x2' : right,
                'y2' : bottom
            })
    
    thresh = 0.3
    
    for stations, coordinate in coordinates.items():
        taken = False
        for detection in detections_bb:
            iou = get_iou(coordinate, detection)
            if iou >= thresh:
                taken = True
                break
        
        if taken or timer[stations][2]:
            ocr_time = get_ocr_datetime(frame)
            if ocr_time is None:
                timer[stations][2] = True
                continue
            else:
                timer[stations][2] = False
                if timer[stations][1] is None:
                    timer[stations][1] = ocr_time
                else:
                    if timer[stations][1] > ocr_time:
                        # invalid time
                        timer[stations][2] = True
                    elif (ocr_time - timer[stations][1]) <= timedelta(seconds=3):
                        timer[stations][0] += (ocr_time - timer[stations][1])
                        timer[stations][1] = ocr_time
                    else:
                        # invalid time
                        timer[stations][2] = True

        color = taken_color if taken else vacant_color
            
        cv2.rectangle(frame, (coordinate['x1'], coordinate['y1']), 
                        (coordinate['x2'], coordinate['y2']), color, 2)
        
        cv2.putText(frame, f"{stations} [{str(timer[stations][0])}]",
                    (coordinate['x1'], coordinate['y1'] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)

    if resize:
        frame = cv2.resize(frame, (width, height), 
                            interpolation=cv2.INTER_AREA)
    img_array.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



# initialize timer per station
# list definition:
# list[0] : total time in work station
# list[1] : number of frames in each work station
timer = {'station_' + str(i): [timedelta(0), 0] for i in range(1,5)}

%cd /content/
cap = cv2.VideoCapture('XVR_ch5_main_20220214100004_20220214110005.mp4')
success,frame = cap.read()

width =  1600
height = 900
resize = False
img_array =[]
for i in tqdm(range(4300)):
    # Capture frame-by-frame
    ret, frame = cap.read()a

    if i <= 2600:
        continue

    detections, width_ratio, height_ratio = darknet_helper(frame, 
                                                           width, 
                                                           height)
    detections = non_max_suppression_fast(detections, 0.65)
    detections_bb = []
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = (int(left * width_ratio), 
                                    int(top * height_ratio), 
                                    int(right * width_ratio), 
                                    int(bottom * height_ratio))
        cv2.rectangle(frame, (left, top), (right, bottom), person_color, 2)
        cv2.putText(frame, "{} [{:.2f}]".format(label, float(confidence)),
                            (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            person_color, 4)
        
        detections_bb.append({
                'x1' : left,
                'y1' : top,
                'x2' : right,
                'y2' : bottom
            })
    
    thresh = 0.2
    
    for stations, coordinate in coordinates.items():
        taken = False
        for detection in detections_bb:
            iou = get_iou(coordinate, detection)
            if iou >= thresh:
                taken = True
                break
        
        if taken:
            timer[stations][1] += 1
            if timer[stations][1] % 15 == 0:
                timer[stations][1] = 0
                timer[stations][0] += timedelta(seconds=1)
              

        color = taken_color if taken else vacant_color
            
        cv2.rectangle(frame, (coordinate['x1'], coordinate['y1']), 
                        (coordinate['x2'], coordinate['y2']), color, 2)
        
        cv2.putText(frame, f"{stations} [{str(timer[stations][0])}]",
                    (coordinate['x1'], coordinate['y1'] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
        
    count_person = len(detections_bb)
    cv2.rectangle(frame, (23, 26), 
                      (208, 63), (0,0,0), -1)
    
    cv2.putText(frame, f"Count of Person: {count_person:0>2}",
                    (23 + 5,26+ 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255, 255), 2)

    if resize:
        frame = cv2.resize(frame, (width, height), 
                            interpolation=cv2.INTER_AREA)
    img_array.append(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



cap = cv2.VideoCapture('resized_cctv_full.mp4')

img_array =[]
success = True
while success:
    success,frame = cap.read()
    # Capture frame-by-frame
    img_array.append(frame)



%cd /content/
fname = 'resized_cctv.mp4'
if not resize:
    width = 1920
    height = 1080
    
if any([True if fname in f else False for f in os.listdir()]):
    !rm resized_cctv.mp4

out = cv2.VideoWriter('/content/resized_cctv.mp4',
                      cv2.VideoWriter_fourcc(*'MP4V'), 
                      20, (1600, 900))

for i in tqdm(range(len(img_array))):
    out.write(img_array[i])
out.release()

%cd darknet

from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = "/content/resized_cctv_full.mp4"

# Compressed video path
compressed_path = "/content/resized_cctv_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=900 height=1600 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)




