from ultralytics import YOLO
# opencv de kullanacaz indir

import cv2 as cv

#modeli yükle

model=YOLO('C:/Users/asus/Desktop/balon_yolo/runs/detect/train4/weights/best.pt')

# canlı video üzerinde tespit yapacağız o yüzden videonun üstünde çalışabilecek döngüyü kur

kamera=cv.VideoCapture(0)

while True:
    isTrue,frame=kamera.read()

    if not (isTrue):
        break


   


kamera.release()



