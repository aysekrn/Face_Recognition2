import cv2
from src.models.face_engine import FaceEngine
from src.models.database_manager import DatabaseManager

def register():
    # Kaydedilecek kişinin adını terminalden al
    name = input("Kaydedilecek kişinin adı: ")
    
    # Motorları başlat
    engine = FaceEngine()
    db = DatabaseManager()
    cap = cv2.VideoCapture(0)

    print("\nKamera açılıyor... Lütfen ekrana bakın ve kaydetmek için klavyeden 's' tuşuna basın.")
    print("Çıkmak için 'q' tuşuna basabilirsiniz.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Ekranda yüzü mavi kare içine alarak gösterelim
        faces = engine.detector.detect(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Hazirsan 's' tusuna bas", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        cv2.imshow("Kayit Ekrani", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            if len(faces) > 0:
                # Ekranda yüz varsa ilk yüzü al ve kaydet
                x, y, w, h = faces[0]
                face_img = frame[y:y+h, x:x+w]
                embedding = engine.model.get_embedding(face_img)
                
                # Veritabanına kaydet
                db.save_face(name, embedding)
                print(f"\n✅ TEBRİKLER! '{name}' başarıyla sisteme kaydedildi!")
                break
            else:
                print("Ekranda net bir yüz bulunamadı, lütfen kameraya doğru bakarak tekrar 's' tuşuna basın.")
        elif key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register()