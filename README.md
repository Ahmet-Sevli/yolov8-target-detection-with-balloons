# 🎈 Real-Time Balloon Detection & Threat Prioritization System

This project is a **real-time object detection and tracking system** developed using **YOLO (Ultralytics)** and **OpenCV**.  
It detects balloons via a camera feed, tracks them across frames, analyzes their movement and size, and determines the **highest-priority threat** based on defined criteria.

---

## 🇹🇷 Türkçe

### 📌 Proje Amacı

Bu projenin amacı, kamera görüntüsü üzerinden balonları **gerçek zamanlı olarak tespit etmek**, takip etmek ve belirlenen kriterlere göre **en yüksek öncelikli hedefi (tehdit)** belirlemektir.

Sistem; nesnelerin:
- Konumunu
- Boyutunu
- Hareket yönünü
- Kameraya göre yaklaşma / uzaklaşma durumunu

analiz ederek dinamik bir tehdit önceliklendirme yapar.

---

### 🧠 Proje Özellikleri

- YOLO modeli ile **nesne tespiti**
- ByteTrack algoritması ile **çoklu nesne takibi**
- Nesne merkez noktası takibi
- Nesne boyutuna göre sınıflandırma (Küçük / Orta / Büyük)
- Hareket yönü tespiti (Sağa, Sola, Yukarı, Aşağı)
- Mesafe değişim analizi (Yaklaşıyor / Uzaklaşıyor)
- Tehdit puanlama ve **en yüksek öncelikli hedefin belirlenmesi**
- Gerçek zamanlı görselleştirme (bounding box, ID, durum bilgileri)

---

### 🧩 Tehdit Önceliklendirme Mantığı

Bir balon için tehdit puanı şu kriterlere göre hesaplanır:

| Kriter | Puan |
|------|------|
| Büyük boyut | +2 |
| Kameraya yaklaşıyor | +2 |
| Merkez bölgede | +1 |

En yüksek toplam puana sahip nesne **öncelikli tehdit** olarak işaretlenir.

---

### 🛠️ Kullanılan Teknolojiler

- Python
- OpenCV
- Ultralytics YOLO
- ByteTrack
- NumPy

---

### 📂 Proje Yapısı

