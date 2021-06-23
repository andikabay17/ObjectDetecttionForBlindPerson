import os
import cv2 
import numpy as np 
import random

class Utils():

    def draw_ped(self, img, label, x0, y0, xt, yt, font_size=0.4, color=(255,127,0), text_color=(255,255,255)):

        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, 1)
        cv2.rectangle(img,
                (x0, y0),  
                (xt, yt), 
                color, 
                2)        
        cv2.rectangle(img,
                        (x0, y0 - h - baseline),  
                        (x0 + w, y0 + baseline), 
                        color, 
                        -1)
        cv2.rectangle(img,
                        (x0, y0 - h - baseline),  
                        (x0 + w, y0 + baseline), 
                        color, 
                        2)  
        cv2.putText(img, 
                    label, 
                    (x0, y0),                   
                    cv2.FONT_HERSHEY_SIMPLEX,     
                    font_size,                          
                    text_color,                
                    1,
                    cv2.LINE_AA) 
        return img

    def postprocess(self, outs, frame, classes, 
                    font_size=0.4, color=(255,127,0), text_color=(255,255,255), 
                    confThreshold = 70, nmsThreshold = 0.3):

        cols, rows = frame.shape[:2]

        classIds = []
        confidences = []
        boxes = []

        for detection in np.array(outs)[0, 0, 0, :, :]:
            confidence = detection[2]*100
            classId = str(int(detection[1]))
            x = detection[3] * cols
            y = detection[4] * rows
            w = (detection[5] * cols) - x
            h = (detection[6] * rows) - y
            classIds.append(classId)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
        
        

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        set_color = [(255, 128, 0), (0, 204, 0), (0, 128, 255)]
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])

            new_color = random.choice(set_color)
            set_color.remove(new_color)

            label_text = "%s (%.2f %%)" % (classes[classIds[i]], confidences[i])
            if classes[classIds[i]] == 'Tembok':
                print(label_text)
            if classes[classIds[i]] == 'Pintu':
                print(label_text)
            if classes[classIds[i]] == 'Meja':
                print(label_text)
            if classes[classIds[i]] == 'Kursi':
                print(label_text)
            frame = self.draw_ped(frame, label_text, x, y, x+w, y+h, 
                            font_size=font_size, 
                            color=color, text_color=text_color)  
        
        
        return frame

   