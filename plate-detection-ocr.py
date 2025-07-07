# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:49:32 2024

@author: Haniyeh
"""
import cv2
from ultralytics import YOLO
import pytesseract 

#load the yolov8 model
model=YOLO('license_plate_detector.pt')

#load the image containing plate
image_path='car3.jpeg'
image=cv2.imread(image_path)

#Run yolov8 on the image
results=model(image)
plate_class_id=0
for result in results:
    for bbox in result.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id=bbox
        if score>0.5 and int(class_id)==plate_class_id:
            cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            #crop plate
            plate_image=image[int(y1):int(y2),int(x1):int(x2)]
            # Display the plate image
            cv2.imwrite('cropped_plate3.jpg', plate_image)
            cv2.imshow('License Plate', plate_image)
            cv2.waitKey(0) 

#OCR
pytesseract.pytesseract.tesseract_cmd=r'C:\Users\V I S I O  N\tesseract\tesseract.exe'

#Use tesseract for doing OCR

plate_text=pytesseract.image_to_string(plate_image,config='--psm 8')

print(f'Detected plate text:{plate_text}')








            
            




