import urllib.request
import os

# Klasörlerin var olduğundan emin olalım
os.makedirs("data/models", exist_ok=True)

print("1. deploy.prototxt indiriliyor (Konfigürasyon dosyası)...")
url_proto = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
urllib.request.urlretrieve(url_proto, "data/models/deploy.prototxt")

print("2. res10_300x300_ssd_iter_140000.caffemodel indiriliyor (Ağırlık dosyası)...")
url_model = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
urllib.request.urlretrieve(url_model, "data/models/res10_300x300_ssd_iter_140000.caffemodel")

print("\nHarika! Tüm dosyalar başarıyla indirildi. Artık python main.py komutunu çalıştırabilirsin. 🚀")