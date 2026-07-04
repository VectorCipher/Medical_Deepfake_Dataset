from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 nano model (it will download automatically)
    model = YOLO('yolov8n.pt')

    # Train the model
    # Note: imgsz=640 is standard, epochs=30 is usually enough for a critic
    model.train(
        data='data.yaml', 
        epochs=30, 
        imgsz=640, 
        project='critic_yolov8', 
        name='weights'
    )

if __name__ == '__main__':
    main()
