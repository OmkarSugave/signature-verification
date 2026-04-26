import os
import shutil

SOURCE = "dataset_raw"
DEST = "dataset/train"

os.makedirs(DEST, exist_ok=True)

files = os.listdir(SOURCE)

for file in files:
    if not file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    name = file.lower()

    # Example naming logic (edit if needed)
    if "g" in name:
        type_ = "genuine"
    elif "f" in name:
        type_ = "forged"
    else:
        continue

    # Extract person id (example: user1_g_1.png)
    person = name.split("_")[0]

    person_path = os.path.join(DEST, person, type_)
    os.makedirs(person_path, exist_ok=True)

    shutil.copy(
        os.path.join(SOURCE, file),
        os.path.join(person_path, file)
    )

print("✅ Dataset reorganized")