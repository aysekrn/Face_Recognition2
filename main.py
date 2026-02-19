import sys
import os

# "src" klasörünün bulunabilmesi için proje kök dizinini sisteme tanıtıyoruz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.MainController import MainController

if __name__ == "__main__":
    app = MainController()
    app.start()