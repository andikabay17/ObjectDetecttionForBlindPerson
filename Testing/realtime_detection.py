import cv2
import json 
import numpy as np
import os

from utils_baru import Utils

utils = Utils()

classesFile = "bisa.json"
with open(classesFile) as json_labels:
    classes = json.load(json_labels)

# parameter
target_w = 750
target_h = 750

# load petrained model (.pb & .pbtxt) faster R-CNN with backbone Resnet 50 on COCO dataset
net = cv2.dnn.readNetFromTensorflow("model/inference_graph_140k/frozen_inference_graph.pb",
                                    "model/inference_graph_140k/opencv_dnn_140k.pbtxt")
image_path = 'hasil_video_23'
destination_path = 'HASIL_DETEKSI_ VIDEO_ 23'

# set CUDA as backend & target OpenCV DNN
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# get output layers
layerOutput = net.getUnconnectedOutLayersNames()

def detect_object(frame):
    blob = cv2.dnn.blobFromImage(frame, size=(target_w,target_h), swapRB=True, crop=False)
    # predict classess & box
    frame=cv2.resize(frame,(target_w,target_h))
    net.setInput(blob)
    output = net.forward(layerOutput)
    
    t, _ = net.getPerfProfile()
   #print('inference time: %.2f s' % (t / cv2.getTickFrequency()))

    return utils.postprocess(output, frame, classes, font_size=0.3, confThreshold=70)

for filename in os.listdir(image_path):
    img = cv2.imread(os.path.join(image_path, filename))
    img = detect_object(img)
#cv2.imshow(filename, img)
    cv2.imwrite(os.path.join(destination_path, filename), img)
    print(filename)
cv2.waitKey(0)
cv2.destroyAllWindows()
