import cv2
import numpy as np
import glob
import os
import shutil
import random
import csv

src_points = np.float32([[0, 390], [640, 390], [640, 190], [0, 190]])
dst_points = np.float32([[0, 480], [640, 480], [640, 0], [0, 0]])
M = cv2.getPerspectiveTransform(src_points, dst_points)

input_base = "sorted_images"
output_bases = ["train_data", "test_data"]
categories = ["forward", "left", "right"]
label_map = {"forward": 0, "left": 1, "right": 2}

for base in output_bases:
    if os.path.exists(base):
        shutil.rmtree(base)
    for cat in categories:
        os.makedirs(os.path.join(base, cat), exist_ok=True)

train_ratio = 0.8


def write_csv(csv_path, data):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label"])
        writer.writerows(data)


def process_and_save(paths, output_folder, cat_name, dataset_name):
    filenames = []
    total = len(paths)
    for idx, path in enumerate(paths, 1):
        frame = cv2.imread(path)
        if frame is None:
            continue
        bev = cv2.warpPerspective(frame, M, (640, 480))
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(output_folder, filename), binary)
        filenames.append(filename)
        print(f"[{dataset_name.upper()} - {cat_name}] {idx}/{total}")
    return filenames


for base in output_bases:
    all_csv_data = []
    for cat in categories:
        input_folder = os.path.join(input_base, cat)
        image_paths = glob.glob(os.path.join(input_folder, "*.*"))
        random.shuffle(image_paths)

        split_idx = int(len(image_paths) * train_ratio)
        if base == "train_data":
            selected_paths = image_paths[:split_idx]
        else:
            selected_paths = image_paths[split_idx:]

        output_folder = os.path.join(base, cat)
        filenames = process_and_save(selected_paths, output_folder, cat, base)

        for fname in filenames:
            all_csv_data.append([f"{cat}/{fname}", label_map[cat]])

    csv_path = f"{base}.csv"
    write_csv(csv_path, all_csv_data)
    print(f"[INFO] {base}.csv saved ({len(all_csv_data)} rows)")
