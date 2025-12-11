import os
import shutil


RAW_TRAIN = "data/raw/VGG2/train"
RAW_TEST = "data/raw/VGG2/test"

PROCESSED = "preprocessing/VGG2"

OUT_TRAIN = "preprocessing/VGG2_train"
OUT_TEST = "preprocessing/VGG2_test"


os.makedirs(OUT_TRAIN, exist_ok=True)
os.makedirs(OUT_TEST, exist_ok=True)

# ----------------------------
train_people = set(os.listdir(RAW_TRAIN))
test_people = set(os.listdir(RAW_TEST))

print(f"Found {len(train_people)} train identities.")
print(f"Found {len(test_people)} test identities.")

# ----------------------------

for person in os.listdir(PROCESSED):
    src_path = os.path.join(PROCESSED, person)

    # skip non-directories
    if not os.path.isdir(src_path):
        continue

    # If person exists in raw/train → move to train output
    if person in train_people:
        dst = os.path.join(OUT_TRAIN, person)
        print(f"[TRAIN] Moving {person}")
    
    # If person exists in raw/test → move to test output
    elif person in test_people:
        dst = os.path.join(OUT_TEST, person)
        print(f"[TEST] Moving {person}")

    else:
        print(f"⚠ WARNING: {person} was not found in raw/train or raw/test. Skipping.")
        continue

    shutil.move(src_path, dst)

print("\n✔ Split completed! Check preprocessing/VGG2_train and preprocessing/VGG2_test")
