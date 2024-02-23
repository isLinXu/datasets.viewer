import os
from collections import defaultdict
import streamlit as st
from tqdm import tqdm


def analyze_yolo_dataset(dataset_path, classes_to_analyze=None, class_names=None, splits=['train', 'test']):
        class_counts = defaultdict(int)
        image_counts = defaultdict(int)
        total_images = 0

        if not os.path.exists(dataset_path):
            st.warning(f"Dataset path '{dataset_path}' does not exist.")
            return

        for split in splits:
            split_dir = os.path.join(dataset_path, 'labels', split)
            if not os.path.exists(split_dir):
                st.warning(f"Split '{split}' does not exist in the dataset.")
                continue

            for label_file in tqdm(os.listdir(split_dir), desc=f"Processing {split} data"):
                with open(os.path.join(dataset_path, 'labels', split, label_file), 'r') as f:
                    lines = f.readlines()

                if classes_to_analyze is None:
                    found_classes = [int(line.split()[0]) for line in lines]
                else:
                    found_classes = [int(line.split()[0]) for line in lines if int(line.split()[0]) in classes_to_analyze]

                for cls in found_classes:
                    class_counts[cls] += 1

                if found_classes:
                    for cls in set(found_classes):
                        image_counts[cls] += 1
                total_images += 1

        # 创建一个字符串来存储分析结果
        analysis_result = ""

        analysis_result += "类别统计：\n"
        for cls, count in class_counts.items():
            class_name = class_names[cls] if class_names else cls
            analysis_result += f"类别 {class_name}: {count} 个标签\n"

        analysis_result += "\n图片统计：\n"
        for cls, count in image_counts.items():
            class_name = class_names[cls] if class_names else cls
            average_labels = class_counts[cls] / count
            analysis_result += f"类别 {class_name}: {count} 张图片 (平均每张图片 {average_labels:.2f} 个标签)\n"

        analysis_result += f"\n总图片数量：{total_images}\n"

        # 将分析结果输出到 text_area 中
        st.text_area("分析结果:", value=analysis_result, height=200)