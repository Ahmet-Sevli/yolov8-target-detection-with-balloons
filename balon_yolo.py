from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch kullanacağımız modeli gösterir yolov8n yani yolo8 nano modeli


# Use the model
results = model.train(data="dataset.yaml", epochs=40)  # train the model

#yolo detect train data=config.yaml model="yolov8n" epochs=1 bu da direkt terminalde çalıştırmak