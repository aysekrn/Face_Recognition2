import cv2
import numpy as np
from src.models.FaceNet import FaceNetModel 
from src.models.face_detector import FaceDetector 
from src.models.database_manager import DatabaseManager

class FaceEngine:
    def __init__(self):
        self.detector = FaceDetector()  
        self.model = FaceNetModel()     
        self.db = DatabaseManager()
        self.known_faces = self.db.load_faces()

    def process_frame(self, frame, threshold=0.6):
        faces = self.detector.detect(frame)
        results = []

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # 128-d Embedding Çıkarımı
                embedding = self.model.get_embedding(face_img)
            except:
                continue
            
            identity = "Bilinmiyor"
            confidence = 0.0

            # Veritabanı ile karşılaştırma
            for name, saved_embedding in self.known_faces.items():
                dist = np.linalg.norm(embedding - saved_embedding)
                if dist < threshold:
                    identity = name
                    confidence = (1 - dist) * 100
                    break
            
            results.append({"box": (x, y, w, h), "name": identity, "score": confidence})
        return results