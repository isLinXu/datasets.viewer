
import os
import streamlit as st
import cv2
from PIL import Image
import random
import json
import numpy as np


class DatasetViewer:
    def __init__(self):
        self.image_folder_path = ""
        self.label_folder_path = ""
        self.class_names = []
        self.colors = []
        self.image_files = []
        self.label_files = []
        self.task_type = ""

    def load_sidebar(self):
        with st.sidebar:
            st.header("设置")
            self.image_folder_path = st.text_input("输入图像文件夹路径:", value="")
            self.label_folder_path = st.text_input("输入标签文件夹路径:", value="")
            class_names = st.text_input("输入类别名称（用逗号分隔）:", value="")
            if class_names:
                self.class_names = [name.strip() for name in class_names.split(",")]
            num_classes = len(self.class_names) if self.class_names else 100
            self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                           range(num_classes)]

            task_options = ["分类", "检测", "分割"]
            self.task_type = st.selectbox("选择任务类型:", task_options)

    def visual_classification(self):
        if self.image_folder_path:
            image_files = os.listdir(self.image_folder_path)
            supported_formats = [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]
            images = [file for file in image_files if any(file.endswith(fmt) for fmt in supported_formats)]

            if not images:
                st.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                return

            image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,
                                                  value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=image_index)

            if image_file:
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)

                if self.label_folder_path:
                    label_files = os.listdir(self.label_folder_path)
                    supported_formats = [".txt", ".json", ".xml"]
                    labels = [file for file in label_files if any(file.endswith(fmt) for fmt in supported_formats)]

                    if not labels:
                        st.warning("标签文件夹中没有找到支持的标签格式。请检查路径和标签格式。")
                        return

                    label_file = os.path.splitext(image_file)[0] + ".txt"
                    if label_file in labels:
                        with open(os.path.join(self.label_folder_path, label_file), "r") as file:
                            content = file.read()

                            st.image(image, caption="src", use_column_width=True)
                            st.write("标签内容:")
                            st.write(content)
                    else:
                        st.warning("未找到对应的标签文件。请确保图像和标签文件具有相同的文件名。")
                else:
                    st.warning("请输入标签文件夹路径。")
        else:
            st.warning("请输入图像文件夹路径。")

    def visual_detection(self):
        if self.image_folder_path:
            image_files = os.listdir(self.image_folder_path)
            supported_formats = [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]
            images = [file for file in image_files if any(file.endswith(fmt) for fmt in supported_formats)]

            if not images:
                st.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                return

            image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,
                                                  value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=image_index)

            if image_file:
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)

                if self.label_folder_path:
                    label_files = os.listdir(self.label_folder_path)
                    supported_formats = [".txt", ".json", ".xml"]
                    annotations = [file for file in label_files if any(file.endswith(fmt) for fmt in supported_formats)]

                    if not annotations:
                        st.warning("标签文件夹中没有找到支持的标注格式。请检查路径和标注格式。")
                        return

                    annotation_file = os.path.splitext(image_file)[0] + ".txt"
                    if annotation_file in annotations:
                        with open(os.path.join(self.label_folder_path, annotation_file), "r") as file:
                            content = file.read()
                            image_cv = cv2.imread(image_path)
                            height, width, _ = image_cv.shape
                            with open(os.path.join(self.label_folder_path, annotation_file), "r") as file:
                                for line in file.readlines():
                                    class_id, x, y, w, h = map(float, line.strip().split())
                                    x_min = int((x - w / 2) * width)
                                    x_max = int((x + w / 2) * width)
                                    y_min = int((y - h / 2) * height)
                                    y_max = int((y + h / 2) * height)
                                    color = self.colors[int(class_id)]
                                    cv2.rectangle(image_cv, (x_min, y_min), (x_max, y_max), color, 2)
                                    if self.class_names:
                                        class_name = self.class_names[int(class_id)]
                                    else:
                                        class_name = str(int(class_id))
                                    cv2.putText(image_cv, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                                col1, col2 = st.columns(2)
                                col1.image(image, caption="src", use_column_width=True)
                                col2.image(image_cv, caption="dst", use_column_width=True)

                                st.write("标注内容:")
                                st.write(content)
                        # else:
                        #     st.warning("未找到对应的标注文件。请确保图像和标注文件具有相同的文件名。")
                    else:
                        st.warning("请输入标签文件夹路径。")
            else:
                st.warning("请输入图像文件夹路径。")

    def visual_segmentation(self):
        if self.image_folder_path:
            image_files = os.listdir(self.image_folder_path)
            supported_formats = [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]
            images = [file for file in image_files if any(file.endswith(fmt) for fmt in supported_formats)]

            if not images:
                st.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                return

            image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,
                                                  value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=image_index)

            if image_file:
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)

                if self.label_folder_path:
                    label_files = os.listdir(self.label_folder_path)
                    supported_formats = [".png", ".bmp", ".tiff"]
                    masks = [file for file in label_files if any(file.endswith(fmt) for fmt in supported_formats)]

                    if not masks:
                        st.warning("标签文件夹中没有找到支持的分割掩码格式。请检查路径和掩码格式。")
                        return

                    mask_file = os.path.splitext(image_file)[0] + ".png"
                    if mask_file in masks:
                        mask_path = os.path.join(self.label_folder_path, mask_file)
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                        if mask is None:
                            st.warning("无法读取分割掩码。请检查掩码文件。")
                            return

                        if mask.shape[0] != image.height or mask.shape[1] != image.width:
                            st.warning("图像和分割掩码的尺寸不匹配。请确保它们具有相同的尺寸。")
                            return

                        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                        for i in range(len(self.colors)):
                            mask_color[mask == i] = self.colors[i]

                        blend = cv2.addWeighted(np.array(image), 0.7, mask_color, 0.3, 0)

                        col1, col2 = st.columns(2)
                        col1.image(image, caption="src", use_column_width=True)
                        col2.image(blend, caption="dst", use_column_width=True)
                    else:
                        st.warning("未找到对应的分割掩码文件。请确保图像和掩码文件具有相同的文件名。")
                else:
                    st.warning("请输入标签文件夹路径。")
        else:
            st.warning("请输入图像文件夹路径。")

    def visual(self):
        if self.task_type == "分类":
            self.visual_classification()
        elif self.task_type == "检测":
            self.visual_detection()
        elif self.task_type == "分割":
            self.visual_segmentation()
        else:
            st.warning("请选择任务类型")

def main():
    st.title("Dataset.Viewer©️(Hertz)")
    viewer = DatasetViewer()
    viewer.load_sidebar()
    viewer.visual()

if __name__ == "__main__":
    main()


