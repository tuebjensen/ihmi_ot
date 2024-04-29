from ultralytics import YOLO

# Load a model
model = YOLO("models/yolov8n.pt")  
model.to('cuda')
batch = 4
workers = 8
save_period = 50
epochs = 300
freeze = 10
cache = True




model.train(data="datasets/gazebo_data.yaml", 
            epochs=epochs, 
            batch=batch, 
            cache=cache, 
            save_period=save_period,
            pretrained=True,
            resume=False,
            freeze=freeze,
            name = f'tennis-{epochs}-{batch}-{workers}')
