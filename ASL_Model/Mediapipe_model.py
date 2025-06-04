import os
import cv2
import csv
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Directories for train and test datasets
TRAIN_DIR = './Datasets/asl_alphabet_train/asl_alphabet_train'
TEST_DIR = './Datasets/asl_alphabet_test/asl_alphabet_test'

# Output CSV files
TRAIN_CSV = 'train_landmarks.csv'
TEST_CSV = 'test_landmarks.csv'

# Letters that involve motion or that you want to skip
SKIP_LETTERS = ['J', 'Z']

# The order of landmarks from MediaPipe Hands (21 landmarks)
# 0: wrist
# 1: thumb_cmc, 2: thumb_mcp, 3: thumb_ip, 4: thumb_tip
# 5: index_mcp, 6: index_pip, 7: index_dip, 8: index_tip
# 9: middle_mcp, 10: middle_pip, 11: middle_dip, 12: middle_tip
# 13: ring_mcp, 14: ring_pip, 15: ring_dip, 16: ring_tip
# 17: pinky_mcp, 18: pinky_pip, 19: pinky_dip, 20: pinky_tip

LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
]

# Generate the header row for CSV
# For each landmark, we have (x,y,z) for world coordinates and (x,y,z) for normalized image coordinates
header = ["Label"]
for landmark_name in LANDMARK_NAMES:
    header.extend([
        f"{landmark_name}_world_x", f"{landmark_name}_world_y", f"{landmark_name}_world_z"
    ])
for landmark_name in LANDMARK_NAMES:
    header.extend([
        f"{landmark_name}_x", f"{landmark_name}_y", f"{landmark_name}_z"
    ])

def get_letter_images(directory):
    """Return a dictionary { 'A': [list_of_image_paths], 'B': [...], ... }."""
    letter_dict = {}
    for letter in os.listdir(directory):
        letter_path = os.path.join(directory, letter)
        if os.path.isdir(letter_path):
            # Get all images for this letter
            images = [os.path.join(letter_path, img) for img in os.listdir(letter_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            letter_dict[letter] = images
    return letter_dict

def process_dataset(data_dict, output_csv):
    """Process a dictionary of letter -> image_paths and write results to CSV."""
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        model_complexity=1
    ) as hands, open(output_csv, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for letter, images in data_dict.items():
            if letter in SKIP_LETTERS:
                # Skip certain letters if needed
                continue
            
            for img_path in images:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                # Convert to RGB and flip horizontally (if desired for consistency)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.flip(image_rgb, 1)
                
                result = hands.process(image_rgb)
                if not result.multi_hand_world_landmarks or not result.multi_hand_landmarks:
                    # If no hand detected, skip
                    continue
                
                world_landmarks = result.multi_hand_world_landmarks[0].landmark
                landmarks = result.multi_hand_landmarks[0].landmark
                
                # Build the row: [Label, all world coords, all normalized coords]
                row = [letter]
                # World coordinates
                for lm in world_landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                # Normalized coordinates
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])
                
                writer.writerow(row)
    print(f"Saved landmarks to {output_csv}")

# Get training and testing images
train_data = get_letter_images(TRAIN_DIR)
test_data = get_letter_images(TEST_DIR)

# Process and save to CSV
process_dataset(train_data, TRAIN_CSV)
process_dataset(test_data, TEST_CSV)
