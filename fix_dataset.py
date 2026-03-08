import os
import shutil

base = "dataset_new"

train_src = os.path.join(base, "train")
test_src = os.path.join(base, "test")

clean_base = "yawn_dataset"

train_dst = os.path.join(clean_base, "train")
val_dst = os.path.join(clean_base, "val")

os.makedirs(train_dst + "/yawn", exist_ok=True)
os.makedirs(train_dst + "/no_yawn", exist_ok=True)

os.makedirs(val_dst + "/yawn", exist_ok=True)
os.makedirs(val_dst + "/no_yawn", exist_ok=True)


def move_images(src, dst, label):

    folder = os.path.join(src, label)

    for file in os.listdir(folder):

        if file.endswith((".jpg", ".png", ".jpeg")):

            shutil.copy(
                os.path.join(folder, file),
                os.path.join(dst, label, file)
            )


move_images(train_src, train_dst, "yawn")
move_images(train_src, train_dst, "no_yawn")

move_images(test_src, val_dst, "yawn")
move_images(test_src, val_dst, "no_yawn")

print("Dataset cleaned successfully")