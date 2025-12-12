import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm


RAW_LFW_PATH = "data/raw/LfW"
RAW_VGG_PATH = "data/raw/VGG2/train"   # لو عايز تشغل على test برضه، هنعملها كمان
RAW_VGG_TEST_PATH = "data/raw/VGG2/test"

OUT_LFW_PATH = "preprocessing/LfW"
OUT_VGG_PATH = "preprocessing/VGG2"
 


detector = MTCNN()

# -------------------------------
def process_face(image):
    results = detector.detect_faces(image)
    if len(results) == 0:
        return None

    x, y, w, h = results[0]['box']

    x = max(0, x)
    y = max(0, y)

    face = image[y:y+h, x:x+w]

    try:
        face = cv2.resize(face, (160, 160))
    except:
        return None

    return face

# -------------------------------

def preprocess_folder(input_path, output_path):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    persons = os.listdir(input_path)

    for person in tqdm(persons, desc=f"Processing {input_path}"):

        person_in = os.path.join(input_path, person)
        person_out = os.path.join(output_path, person)

        if not os.path.isdir(person_in):
            continue

        os.makedirs(person_out, exist_ok=True)

        for img_name in os.listdir(person_in):
            img_path = os.path.join(person_in, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            processed = process_face(img)

            if processed is not None:
                out_path = os.path.join(person_out, img_name)
                cv2.imwrite(out_path, processed)

# -------------------------------

print("Starting LFW preprocessing...")
preprocess_folder(RAW_LFW_PATH, OUT_LFW_PATH)

print("Starting VGG2 Train preprocessing...")
preprocess_folder(RAW_VGG_PATH, OUT_VGG_PATH)

print("Starting VGG2 Test preprocessing...")
preprocess_folder(RAW_VGG_TEST_PATH, OUT_VGG_PATH)

print("DONE! All preprocessing completed successfully.")
