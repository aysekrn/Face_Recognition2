import cv2

class WebcamView:
    def display(self, frame, results):
        for res in results:
            x, y, w, h = res["box"]
            label = f"{res['name']} ({res['score']:.1f}%)"
            color = (0, 255, 0) if res['name'] != "Bilinmiyor" else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow("Profesyonel Yuz Tanima Sistemi", frame)

    def close(self):
        cv2.destroyAllWindows()