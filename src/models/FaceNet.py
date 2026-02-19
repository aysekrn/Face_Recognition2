import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1
from PIL import Image

class FaceNetModel:
    def __init__(self):
        # Ekran kartı (GPU) varsa onu, yoksa işlemciyi (CPU) kullanır
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"FaceNet Modeli Yükleniyor... Donanım: {self.device}")
        
        # Sadece öznitelik çıkarıcıyı (InceptionResnetV1) yüklüyoruz.
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def get_embedding(self, face_img):
        """OpenCV'den gelen yüz resmini 512 boyutlu vektöre çevirir."""
        # 1. BGR (OpenCV) formatından RGB formatına çevir
        rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # 2. Resmi PIL formatına çevir ve FaceNet'in beklediği 160x160 boyutuna getir
        pil_img = Image.fromarray(rgb_img).resize((160, 160))
        
        # 3. Resmi PyTorch tensörüne çevir ve normalize et
        img_tensor = torch.tensor(np.array(pil_img)).permute(2, 0, 1).float()
        img_tensor = (img_tensor - 127.5) / 128.0 
        
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # 4. Vektörü (Embedding) Çıkar
        with torch.no_grad():
            embedding = self.resnet(img_tensor)
        
        return embedding.detach().cpu().numpy()[0]