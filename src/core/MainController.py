import cv2
from src.models.face_engine import FaceEngine
from src.views.webcam_view import WebcamView

class MainController:
    def __init__(self):
        self.engine = FaceEngine()
        self.view = WebcamView()
        self.camera = cv2.VideoCapture(0)

    def start(self):
        print("Sistem başlatıldı. Çıkmak için 'q' tuşuna basın.")
        while True:
            success, frame = self.camera.read()
            if not success: 
                break

            # Model işleme
            results = self.engine.process_frame(frame)
            
            # View görselleştirme
            self.view.display(frame, results)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.camera.release()
        self.view.close()