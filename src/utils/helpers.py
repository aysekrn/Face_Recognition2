import os
import cv2

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_face(face_img, target_size=(160, 160)):
    if face_img is None or face_img.size == 0:
        return None
    return cv2.resize(face_img, target_size)