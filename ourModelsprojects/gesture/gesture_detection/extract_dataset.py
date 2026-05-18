import os
import cv2
import csv
import copy
import itertools
import mediapipe as mp

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, _ in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def main():
    # Settings
    dataset_path = 'dataset/extracted/hagrid-sample-30k-384p/hagrid_30k'
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    
    # Mapping (Folder Name: Label ID)
    # 0: Open, 1: Close, 2: Pointer, 3: OK
    mapping = {
        'train_val_palm': 0,
        'train_val_fist': 1,
        'train_val_one': 2,
        'train_val_ok': 3
    }

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )

    # Open CSV for writing (Clear it first or append? We'll overwrite to keep it clean)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        for folder_name, label in mapping.items():
            full_folder_path = os.path.join(dataset_path, folder_name)
            if not os.path.exists(full_folder_path):
                print(f"Warning: {full_folder_path} not found.")
                continue
            
            print(f"Processing {folder_name} (Label {label})...")
            images = [f for f in os.listdir(full_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            count = 0
            for image_name in images:
                image_path = os.path.join(full_folder_path, image_name)
                image = cv2.imread(image_path)
                if image is None: continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract Landmark coordinates
                        landmark_list = []
                        for landmark in hand_landmarks.landmark:
                            landmark_list.append([landmark.x, landmark.y])
                        
                        # Pre-process
                        processed_landmarks = pre_process_landmark(landmark_list)
                        
                        # Write to CSV
                        writer.writerow([label, *processed_landmarks])
                        count += 1
                
                if count >= 1000: # We limit to 1000 images per class for faster initial training
                    break
            
            print(f"Done. Extracted {count} samples for label {label}.")

    hands.close()
    print(f"Extraction complete! Data saved to {csv_path}")

if __name__ == '__main__':
    main()
