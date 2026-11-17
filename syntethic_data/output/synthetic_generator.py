"""
Synthetic Data Generation Pipeline
Generates augmented dataset images by compositing foreground objects (motorcycles) 
onto random background scenes and automatically calculating YOLO bounding box annotations.
"""

import cv2
import numpy as np
import random
import os
from PIL import Image

def load_background(path):
    background = cv2.imread(path)
    if background is None:
        print(f"Error: Background not found at {path}")
        return None
    return background

def load_motorcycle(path):
    # cv2.IMREAD_UNCHANGED is crucial to retain the 4th Alpha channel (transparency) from PNG files
    motorcycle = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if motorcycle is None:
        print(f"Error: Motorcycle not found at {path}")
        return None
    return motorcycle

def get_image_info(image):
    return image.shape

def resize_motorcycle(motorcycle, bg_width, bg_height):
    height, width = motorcycle.shape[:2]

    # Constrain the object to occupy a maximum of 50% of the background dimensions
    max_width = int(bg_width * 0.5)
    max_height = int(bg_height * 0.5)

    # Calculate the maximum allowable scale to prevent out-of-bounds errors
    max_scale = min(max_width / width, max_height / height)

    # Randomize scale to simulate varying distances from the camera (helps with Small Object Detection)
    scale = random.uniform(max_scale * 0.5, max_scale)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(motorcycle, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized

def generate_random_position(bg_width, bg_height, obj_width, obj_height):
    max_acceptable_x = bg_width - obj_width
    max_acceptable_y = bg_height - obj_height

    # Define a 'horizon' threshold (30% from the top) to prevent vehicles from spawning in the sky
    min_acceptable_y = int(bg_height * 0.3)

    if min_acceptable_y > max_acceptable_y:
        raise ValueError("Object is too large for the specified background constraints!")

    x_offset = random.randint(0, max_acceptable_x)
    y_offset = random.randint(min_acceptable_y, max_acceptable_y)

    return x_offset, y_offset
        
def composite_images(background, motorcycle, x_offset, y_offset):
    # Convert color spaces: OpenCV uses BGR natively, while PIL uses RGB
    background_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    motorcycle_rgba = cv2.cvtColor(motorcycle, cv2.COLOR_BGRA2RGBA)

    background_pil = Image.fromarray(background_rgb)
    motorcycle_pil = Image.fromarray(motorcycle_rgba)

    # Extract the Alpha mask for seamless edge blending
    alpha = motorcycle_pil.getchannel('A')

    # Paste the foreground onto the background using the alpha mask
    background_pil.paste(motorcycle_pil, (x_offset, y_offset), mask=alpha)

    # Convert back to NumPy array and OpenCV BGR format for saving
    final_image = np.array(background_pil)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)

    return final_image

def generate_yolo_annotation(x, y, obj_width, obj_height, image_width, image_height, class_id=3):
    """
    Converts absolute pixel coordinates into YOLO normalized format (0.0 to 1.0).
    Note: class_id=3 typically corresponds to 'motorcycle' in standard vehicle datasets.
    """
    normalized_obj_width = obj_width / image_width
    normalized_obj_height = obj_height / image_height

    # Calculate the normalized center point of the bounding box
    x_center = (x + (obj_width / 2)) / image_width
    y_center = (y + (obj_height / 2)) / image_height
    
    return f'{class_id} {x_center:.6f} {y_center:.6f} {normalized_obj_width:.6f} {normalized_obj_height:.6f}'

def save_result(image, annotation, image_path, label_path):
    cv2.imwrite(image_path, image)
    with open(label_path, 'w') as file:
        file.write(annotation)

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================

NUM_IMAGES_TO_GENERATE = 40

bg_path = "../syntethic_data/background"
moto_path = "../syntethic_data/motorcycles"

# Optimization: Read directory contents once before the loop to reduce I/O overhead
bg_files = os.listdir(bg_path)
moto_files = os.listdir(moto_path)

for idx in range(NUM_IMAGES_TO_GENERATE):
    
    # Randomly sample assets for the current synthetic frame
    bg_file = random.choice(bg_files)
    bg_absolute_path = os.path.join(bg_path, bg_file)
    bg = load_background(bg_absolute_path)

    moto_file = random.choice(moto_files)
    moto_absolute_path = os.path.join(moto_path, moto_file)
    moto = load_motorcycle(moto_absolute_path)

    if bg is None or moto is None:
        continue

    bg_height, bg_width = bg.shape[:2]

    # 1. Scale the object dynamically based on background resolution
    moto_resized = resize_motorcycle(moto, bg_width, bg_height)
    obj_height, obj_width = moto_resized.shape[:2]

    if obj_width > bg_width or obj_height > bg_height:
        continue

    # 2. Determine safe insertion coordinates (avoiding sky/out-of-bounds)
    x_offset, y_offset = generate_random_position(bg_width, bg_height, obj_width, obj_height)

    # 3. Perform alpha-compositing
    final_image = composite_images(bg, moto_resized, x_offset, y_offset)

    # 4. Generate corresponding Ground Truth annotation
    annotation = generate_yolo_annotation(x_offset, y_offset, obj_width, obj_height, bg_width, bg_height)

    # 5. Export results
    image_path = f"../syntethic_data/output/synthetic_{idx + 1:04d}.jpg"
    label_path = f"../syntethic_data/output/synthetic_{idx + 1:04d}.txt"

    save_result(final_image, annotation, image_path, label_path)

    print(f"Generated synthetic sample {idx + 1}/{NUM_IMAGES_TO_GENERATE}")