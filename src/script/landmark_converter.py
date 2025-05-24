import os
import csv
import json
import mediapipe as mp
from tqdm import tqdm
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dataset_dir = 'src/data/processed/augmented_plus_II'
metadata_path = os.path.join(dataset_dir, 'metadata.json')
output_dir = 'src/data/landmarked'
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, 'landmarked_dataset_plus.csv')

# === Carica metadata ===
with open(metadata_path, 'r') as f:
    metadata_list = json.load(f)

metadata_dict = {
    os.path.join(dataset_dir, entry['filename']): {
        'hand_side': entry['hand_side'],
        'source_id': entry['source_id']
    }
    for entry in metadata_list
}

# === MediaPipe Initialization ===
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

# === Creazione CSV con landmark e metadati ===
classes = [cls for cls in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, cls))]

with open(csv_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    header = ['letter', 'filename', 'hand_side', 'source_id']
    for i in range(21):
        header.extend([f'landmark_{i}_x', f'landmark_{i}_y', f'landmark_{i}_z'])
    csv_writer.writerow(header)

    for cls in tqdm(classes, desc="Elaborazione classi"):
        cls_dir = os.path.join(dataset_dir, cls)
        images = [os.path.join(cls_dir, img) for img in os.listdir(cls_dir)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_path in images:
            try:
                image = mp.Image.create_from_file(img_path)
                detection_result = detector.detect(image)
            except:
                continue

            if not detection_result.hand_landmarks:
                continue

            for hand_landmarks in detection_result.hand_landmarks:
                meta = metadata_dict.get(img_path, {'hand_side': 'unknown', 'source_id': 'unknown'})
                row = [cls, img_path, meta['hand_side'], meta['source_id']]
                for landmark in hand_landmarks:
                    row.extend([landmark.x, landmark.y, landmark.z])
                csv_writer.writerow(row)

print(f"CSV file successfully saved to: {csv_path}")