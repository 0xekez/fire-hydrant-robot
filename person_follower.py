import os
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import sys
import RPi.GPIO as GPIO
from time import sleep, time
import label_map_util

def set_angle(angle):
    duty = angle/18+2.5
    # turn it on
    GPIO.output(3, True)
    # give it an angle
    pwm.ChangeDutyCycle(duty)
def stop_servo():
    GPIO.output(3, False)
    pwm.ChangeDutyCycle(0)

# Set up camera constants
IM_WIDTH = 640    #Use smaller resolution for
IM_HEIGHT = 480   #slightly faster framerate
# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
# Grab path to current working directory
CWD_PATH = os.getcwd()
# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 90

##########
# SERVO
##########
SERVO_STEP = 5
DISTANCE_THRESHOLD = 0.05
# set naming for pins
GPIO.setmode(GPIO.BOARD)
# make pin 3 output PWN
GPIO.setup(3, GPIO.OUT)
# set pin 3 PWN to 5 hz
pwm = GPIO.PWM(3, 50)
# set the servo to 0 degrees
pwm.start(0)
current_angle = 90
set_angle(current_angle)
sleep(2)
stop_servo()

print("loading label map & model")
# Load the label map.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
print("loaded")
# Define input and output tensors (i.e. data) for the object detection classifier
# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize camera
camera = PiCamera(resolution=(IM_WIDTH,IM_HEIGHT))
# camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 2
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

'''
a note on boxes:
the boxes are normalized. that is to say that they are numbers between 0 and 1
that tell their locations relative to the image's size.

to convert to pixel locations do the following:
ymin, xmin, ymax, xmax = box
(left, right, top, bottom) = (xmin*im_width, xmax*im_width,
                              ymin*imheight, ymax*im_height)
the four corners cords are given by:
(left, top) (left, bottom) (right, top) (right, bottom)
'''
print("entering main loop")
for image in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    stop_servo()
    start = time()
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    frame = image.array
    frame.setflags(write=1)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    scores = np.squeeze(scores)
    boxes = np.squeeze(boxes)
    classes = np.squeeze(classes)
    people = []
    for i in range(boxes.shape[0]):
        if scores[i] > 0.4:
            if classes[i] in category_index.keys():
                what = category_index[classes[i]]['name']
                if what == 'person':
                        # check what side of the center the person is
                        _, xmin, _, xmax = boxes[i]
                        # these are values ranging from 0 to 1
                        # take the average: this is the cetner of the box
                        average = (xmin+xmax)/2
                        people.append(average)

    # choose person closest to center (least work to move to)
    if  people:
        person = min(people, key=lambda a: abs(a-0.5))
        print("{}\nchoosing:\t{}".format(people, person))
        '''
        now that we have a person, we need to make our movement
        movement should add A NUMBER to the current_angle depending
        on where the person was found
        current_angle should be locked between 0 and 180 degrees
        movement should only happen if the center of the person is more than
        A PERCENTAGE of the screen away from the center
        '''
        distance = abs(person-0.5)
        if distance > DISTANCE_THRESHOLD:
            step_mod = 2*SERVO_STEP*distance
            if person > 0.5:
                current_angle += SERVO_STEP + step_mod
            else:
                current_angle -= SERVO_STEP + step_mod
            if current_angle > 180:
                current_angle = 180
            if current_angle < 0:
                current_angle = 0
            set_angle(current_angle)
    rawCapture.truncate(0)
