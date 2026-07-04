import os
from ultralytics import YOLO
from config import IMAGES_DIR, CRITIC_LABELS_DIR

def setup_yolo_dataset():
    """
    YOLO strictly requires the 'labels' folder to be exactly next to the 'images' folder.
    Since IMAGES_DIR is read-only on Kaggle, we create a fake dataset folder using symlinks.
    """
    dataset_dir = "/kaggle/working/dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    images_link = os.path.join(dataset_dir, "images")
    labels_link = os.path.join(dataset_dir, "labels")
    
    if not os.path.exists(images_link):
        os.symlink(IMAGES_DIR, images_link)
        
    if not os.path.exists(labels_link):
        os.symlink(CRITIC_LABELS_DIR, labels_link)
        
    print(f"Created YOLO dataset structure at {dataset_dir}")

def main():
    setup_yolo_dataset()
    
    # Load a pretrained YOLOv8 nano model (it will download automatically)
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(
        data='data.yml', 
        epochs=30, 
        imgsz=640, 
        project='critic_yolov8', 
        name='weights'
    )

if __name__ == '__main__':
    main()
