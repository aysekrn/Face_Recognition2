import cv2
import os
import numpy as np
from src.utils.helpers import get_project_root

class FaceDetector:
    def __init__(self):
        root_dir = get_project_root()
        prototxt = os.path.join(root_dir, "data", "models", "deploy.prototxt")
        model = os.path.join(root_dir, "data", "models", "res10_300x300_ssd_iter_140000.caffemodel")
        
        # Derin öğrenme tabanlı OpenCV yüz tespit modelini yüklüyoruz
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def detect(self, frame, conf_threshold=0.6):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Koordinatların sınırları aşmasını engelliyoruz
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                if endX - startX > 0 and endY - startY > 0:
                    faces.append((startX, startY, endX - startX, endY - startY))
        return faces