import os
import shutil
import random

SOURCE = "dataset_raw"
DEST = "dataset"

train_ratio = 0.8

users = os.listdir(SOURCE)
random.shuffle(users)

split_index = int(len(users) * train_ratio)

train_users = users[:split_index]
test_users = users[split_index:]

def copy_users(user_list, target):
    for user in user_list:
        src_path = os.path.join(SOURCE, user)
        dst_path = os.path.join(DEST, target, user)
        shutil.copytree(src_path, dst_path)

copy_users(train_users, "train")
copy_users(test_users, "test")

print("Dataset split complete (NO data leakage)")