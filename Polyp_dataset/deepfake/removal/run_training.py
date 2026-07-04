import os
import shutil
from ultralytics import YOLO
from config import IMAGES_DIR, CRITIC_LABELS_DIR

def setup_yolo_dataset():
    """
    YOLO strictly requires the 'labels' folder to be exactly next to the 'images' folder.
    Also, YOLO internally resolves symlinks to their absolute paths, which breaks things
    if the original files are in a read-only directory (like Kaggle input).
    So, we simply copy the files to the working directory.
    """
    dataset_dir = "/kaggle/working/dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    
    images_dest = os.path.join(dataset_dir, "images")
    labels_dest = os.path.join(dataset_dir, "labels")
    
    if not os.path.exists(images_dest):
        print("Copying images to working directory (YOLO requirement)...")
        shutil.copytree(IMAGES_DIR, images_dest)
        
    if not os.path.exists(labels_dest):
        print("Copying labels to working directory...")
        shutil.copytree(CRITIC_LABELS_DIR, labels_dest)
        
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
