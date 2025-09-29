import cv2 as cv
from ultralytics import YOLO
import numpy as np

# YOLO modelini yükle (kendi yolunu değiştir)
model = YOLO('C:/Users/asus/Desktop/balon_yolo/runs/detect/train4/weights/best.pt')




# Kamera veya video kaynağı (0 = webcam)
kamera = cv.VideoCapture(0)

# Önceki kare bilgilerini saklamak için   "C:/Users/asus/Desktop/balon.mp4"
onceki_merkezler = {}
onceki_alanlar = {}

def konum_bul(cx, cy, genislik, yukseklik):
    pos_x = "Sol" if cx < genislik//3 else "Sağ" if cx > 2*genislik//3 else "Merkez"
    pos_y = "Üst" if cy < yukseklik//3 else "Alt" if cy > 2*yukseklik//3 else "Merkez"
    if pos_x == "Merkez" and pos_y == "Merkez":
        return "Merkez"
    return f"{pos_x}-{pos_y}"

def boyut_bul(alan):
    if alan < 5000:
        return "Küçük"
    elif alan < 15000:
        return "Orta"
    else:
        return "Büyük"

def hareket_bul(onceki, simdiki):
    dx, dy = simdiki[0] - onceki[0], simdiki[1] - onceki[1]
    if abs(dx) < 5 and abs(dy) < 5:
        return "Sabit"
    yon_x = "Sağa" if dx > 0 else "Sola" if dx < 0 else ""
    yon_y = "Aşağı" if dy > 0 else "Yukarı" if dy < 0 else ""
    return f"{yon_x} {yon_y}".strip()

def mesafe_bul(onceki_alan, simdiki_alan):
    if onceki_alan is None:
        return "Bilinmiyor"
    if simdiki_alan > onceki_alan * 1.1:
        return "Yaklaşıyor"
    elif simdiki_alan < onceki_alan * 0.9:
        return "Uzaklaşıyor"
    else:
        return "Sabit Mesafe"

# Metin stilleri
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICK = 2

while True:
    ret, kare = kamera.read()
    if not ret:
        break

    yukseklik, genislik = kare.shape[:2]

    # YOLO + ByteTrack ile takip
    sonuc = model.track(kare, persist=True, tracker="bytetrack.yaml")

    balonlar = []
    # Eğer tespit yoksa sonuc[0].boxes.id None olabilir
    if len(sonuc) > 0 and getattr(sonuc[0].boxes, "id", None) is not None:
        for kutu, takip_id in zip(sonuc[0].boxes.xyxy, sonuc[0].boxes.id):
            x1, y1, x2, y2 = map(int, kutu)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            alan = (x2 - x1) * (y2 - y1)
            bid = int(takip_id)

            konum = konum_bul(cx, cy, genislik, yukseklik)
            boyut = boyut_bul(alan)
            hareket = hareket_bul(onceki_merkezler[bid], (cx, cy)) if bid in onceki_merkezler else "Bilinmiyor"
            mesafe = mesafe_bul(onceki_alanlar.get(bid, None), alan)

            balonlar.append({
                "id": bid,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "merkez": (cx, cy),
                "konum": konum,
                "boyut": boyut,
                "hareket": hareket,
                "mesafe": mesafe,
                "alan": alan
            })

            # Geçmişi güncelle (çizimdan önce güncellemek, bir sonraki kare için referans sağlar)
            onceki_merkezler[bid] = (cx, cy)
            onceki_alanlar[bid] = alan

    # Tehdit önceliği hesaplama
    tehdit = None
    max_puan = -1
    for b in balonlar:
        puan = 0
        if b["boyut"] == "Büyük": puan += 2
        if b["mesafe"] == "Yaklaşıyor": puan += 2
        if b["konum"] == "Merkez": puan += 1
        if puan > max_puan:
            max_puan = puan
            tehdit = b

    # Görselleştirme: önce tüm kutuları çiz ama renk tehdide göre ayarla
    for b in balonlar:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cid = b["id"]

        # Tehditse kırmızı, değilse sarı (BGR)
        if tehdit is not None and cid == tehdit["id"]:
            kutu_rengi = (0, 0, 255)   # kırmızı
            metin_rengi = (0, 0, 255)
        else:
            kutu_rengi = (0, 255, 255) # sarı
            metin_rengi = (0, 255, 255)

        # Kutu çiz
        cv.rectangle(kare, (x1, y1), (x2, y2), kutu_rengi, 2)

        # ID metnini kutu üstüne yaz (arka planlı)
        metin = f"Balon {cid}"
        (text_w, text_h), baseline = cv.getTextSize(metin, FONT, FONT_SCALE, THICK)
        # metin arka plan dikdörtgeni
        rect_pt1 = (x1, y1 - text_h - baseline - 6)
        rect_pt2 = (x1 + text_w + 6, y1)
        # sınırların ekran dışına çıkmasını engelle
        rect_pt1 = (max(rect_pt1[0], 0), max(rect_pt1[1], 0))
        rect_pt2 = (min(rect_pt2[0], genislik), min(rect_pt2[1], yukseklik))
        cv.rectangle(kare, rect_pt1, rect_pt2, (0, 0, 0), cv.FILLED)  # siyah arka plan
        text_org = (rect_pt1[0] + 3, rect_pt2[1] - baseline - 2)
        cv.putText(kare, metin, text_org, FONT, FONT_SCALE, metin_rengi, THICK)

        # Merkez noktası
        cv.circle(kare, b["merkez"], 4, (255, 0, 0), -1)

        # İsteğe bağlı: küçük bilgi satırı kutunun altına yaz
        info = f"{b['mesafe']} | {b['boyut']} | {b['hareket']}"
        (iw, ih), ib = cv.getTextSize(info, FONT, 0.5, 1)
        info_org = (x1, y2 + ih + 6)
        if info_org[1] < yukseklik:
            cv.rectangle(kare, (info_org[0]-2, y2+2), (info_org[0]+iw+4, y2+ih+6), (0,0,0), cv.FILLED)
            cv.putText(kare, info, (info_org[0], y2+ih+4), FONT, 0.5, (255,255,255), 1)

    # Konsola da bilgi yazdır (isteğe bağlı, uzunluğu azaltıldı)
    for b in balonlar:
        print(f"Balon ID:{b['id']} Konum:{b['konum']} Durum:{b['mesafe']} Boyut:{b['boyut']} Hareket:{b['hareket']}")

    if tehdit:
        print(f"--- En Yüksek Öncelikli Tehdit --- Hedef ID: {tehdit['id']} ({tehdit['boyut']} ve {tehdit['mesafe']}, {tehdit['konum']})")

    cv.imshow("Balon Tespiti", kare)

    # Çıkış: 'q' veya ESC
    key = cv.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

kamera.release()
cv.destroyAllWindows()
