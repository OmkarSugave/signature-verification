import os
import shutil
import random

SOURCE = "dataset/train"
DEST = "dataset/test"

os.makedirs(DEST, exist_ok=True)

for person in os.listdir(SOURCE):
    src_person = os.path.join(SOURCE, person)
    dst_person = os.path.join(DEST, person)

    for type_ in ["genuine", "forged"]:
        src_type = os.path.join(src_person, type_)
        dst_type = os.path.join(dst_person, type_)

        os.makedirs(dst_type, exist_ok=True)

        images = os.listdir(src_type)
        random.shuffle(images)

        split = int(0.2 * len(images))  # 20% test

        for img in images[:split]:
            shutil.move(
                os.path.join(src_type, img),
                os.path.join(dst_type, img)
            )

print("✅ Test dataset created")