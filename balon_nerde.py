from ultralytics import YOLO
import cv2

# 1️⃣ Eğitilmiş modeli yükle
model = YOLO("C:/Users/asus/Desktop/balon_yolo/runs/detect/train4/weights/best.pt")  # senin yolun



# 2️⃣ Test resmi veya kamerayı aç
image_path = "C:/Users/asus/Desktop/resim3.jpg"  # test resmi  "C:\Users\asus\Desktop\download.jpg"
image = cv2.imread(image_path)
height, width, _ = image.shape
center_x = width / 2
center_y = height / 2





# 3️⃣ Tahmin yap
results = model(image_path)

# 4️⃣ Sonuçları kontrol et
for result in results:
    boxes = result.boxes  # YOLO kutuları
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # sol üst ve sağ alt köşe
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2

        # 5️⃣ Konumu belirle
        if box_center_x < center_x - width*0.1:
            horizontal = "Sol"
        elif box_center_x > center_x + width*0.1:
            horizontal = "Sag"
        else:
            horizontal = "Ortada"

        if box_center_y < center_y - height*0.1:
            vertical = "Ust"
        elif box_center_y > center_y + height*0.1:
            vertical = "Alt"
        else:
            vertical = "Ortada"

        konum = f"{vertical}-{horizontal}"
        print(f"Balon koordinatlari: ({x1},{y1},{x2},{y2}) → Konum: {konum}")

        # 6️⃣ Görselleştirme (isteğe bağlı)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, konum, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

# 7️⃣ Sonucu göster
cv2.imshow("Balon Tespiti", image)
cv2.waitKey(0)
cv2.destroyAllWindows()