import os
import csv
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dataset_dir = 'src/data/LIS-fingerspelling-dataset'
output_dir = 'src/data/landmarked'  # Directory dove salvare il CSV
os.makedirs(output_dir, exist_ok=True)
csv_path = 'landmarked_dataset.csv'  # Percorso completo del CSV

# === MediaPipe Initialization ===
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a hand landmarker instance with the image mode:
base_options = python.BaseOptions(model_asset_path='/home/acaia/LIS/LIS-Gesture-Translator/hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  # Rileva fino a 2 mani
    running_mode=VisionRunningMode.IMAGE  # Modalità per immagini statiche
)
detector = vision.HandLandmarker.create_from_options(options)

# === Creating CSV File From Landmark MediaPipe === 
classes = [cls for cls in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cls))]

with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Creating csv file header
    header = ['letter']
    for i in range(21):  # 21 landmarks * 3 (x,y,z) dim for a hand
        header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
    csv_writer.writerow(header)  # Scrivi l'header una sola volta

    for cls in tqdm(classes, desc="Elaborazione classi"):
        cls_dir = os.path.join(dataset_dir, cls)
        images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir) 
                 if img.lower().endswith(('.png', '.jpg', '.jpeg'))]  # Filtra solo immagini

        for img_path in images:
            # Load the input image
            image = mp.Image.create_from_file(img_path)

            # Detect hand landmarks
            detection_result = detector.detect(image)

            # Process only if hands are detected
            if not detection_result.hand_landmarks:
                continue  # Skip images with no detected hands

            # For each detected hand (up to 2)
            for hand_landmarks in detection_result.hand_landmarks:
                row = [cls]
                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])
                csv_writer.writerow(row)

print(f"CSV file successfully saved to: {csv_path}")