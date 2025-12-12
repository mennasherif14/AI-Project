import os
import cv2
import albumentations as A
from tqdm import tqdm


# INPUT FOLDERS (بعد preprocessing)
# -----------------------------------
INPUT_FOLDERS = [
    "preprocessing/LfW",
    "preprocessing/VGG2_train",
    "preprocessing/VGG2_test"
]


# OUTPUT BASE DIRECTORY
# -----------------------------------
OUTPUT_BASE = "augmented_fast"


# Albumentations Augmentation Pipeline
# -----------------------------------
transform = A.Compose([
    A.Rotate(limit=15, p=0.7),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.7),
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.4),
    A.GaussianBlur(blur_limit=3, p=0.3)
])

# Make sure output folders exist
# -----------------------------------
for folder in INPUT_FOLDERS:
    out_path = os.path.join(OUTPUT_BASE, os.path.basename(folder))
    os.makedirs(out_path, exist_ok=True)

# -----------------------------------
# Augment a single person folder
# -----------------------------------
def augment_person(input_folder, output_folder, n_aug=5):

    os.makedirs(output_folder, exist_ok=True)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # Create n_aug augmented images
        for i in range(n_aug):
            augmented = transform(image=img)["image"]
            save_name = f"aug_{i}_{img_name}"
            cv2.imwrite(os.path.join(output_folder, save_name), augmented)

# -----------------------------------
# Main loop

for dataset in INPUT_FOLDERS:

    print(f"\n Starting fast augmentation for: {dataset}")

    dataset_name = os.path.basename(dataset)
    output_dataset_folder = os.path.join(OUTPUT_BASE, dataset_name)

    persons = os.listdir(dataset)

    for person in tqdm(persons, desc=f"Augmenting {dataset_name}"):

        input_person = os.path.join(dataset, person)
        output_person = os.path.join(output_dataset_folder, person)

        if os.path.isdir(input_person):
            augment_person(input_person, output_person, n_aug=5)

print("\n FAST AUGMENTATION COMPLETED SUCCESSFULLY!")
