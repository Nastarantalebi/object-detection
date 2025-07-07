import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np

model = YOLO("license_plate_detector.pt")

image_path = 'car.jpeg'
image = cv2.imread(image_path)

results = model(image)
plate_class_id = 0
plate_found = False

for result in results:
    for bbox in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = bbox
        if score > 0.5 and int(class_id) == plate_class_id:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            plate_image = image[int(y1):int(y2), int(x1):int(x2)]
            cv2.imwrite('cropped_plate.jpeg', plate_image)
            cv2.imshow('license_plate', plate_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            plate_found = True
            break
    if plate_found:
        break

if not plate_found:
    print("No plate detected!")
    exit()

# OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

cv2.imwrite('thresh_plate.jpeg', thresh)

plate_text = pytesseract.image_to_string(thresh, config='--psm 8')
print(f'Detected plate text: {plate_text}')
