import json
import os
import numpy as np
from src.utils.helpers import get_project_root

class DatabaseManager:
    def __init__(self):
        self.db_path = os.path.join(get_project_root(), "data", "known_faces.json")
        # Dosya hiç yoksa otomatik oluşturur
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w") as f:
                json.dump({}, f)

    def load_faces(self):
        if not os.path.exists(self.db_path):
            return {}
        
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
                return {name: np.array(vec) for name, vec in data.items()}
        # Dosya tamamen boşsa veya bozuksa çökmeyi engeller, boş sözlük döner
        except json.JSONDecodeError: 
            return {}

    def save_face(self, name, embedding):
        faces = self.load_faces()
        faces[name] = embedding
        serializable = {k: v.tolist() for k, v in faces.items()}
        with open(self.db_path, "w") as f:
            json.dump(serializable, f, indent=4)