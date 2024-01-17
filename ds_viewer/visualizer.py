
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
        self.image_index = 0

    def get_files(self, folder_path, supported_formats):
        files = os.listdir(folder_path)
        return [file for file in files if any(file.endswith(fmt) for fmt in supported_formats)]

    def load_sidebar(self):
        '''
        加载侧边栏
        :return:
        '''
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

    def draw_mask(self, image, mask, colors):
        mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for i in range(len(colors)):
            mask_color[mask == i] = colors[i]
        blend = cv2.addWeighted(np.array(image), 0.7, mask_color, 0.3, 0)
        return blend

    def visual_classification(self):
        '''
        可视化分类
        :return:
        '''
        if self.image_folder_path:
            images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
            if not images:
                st.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                return

            self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=self.image_index)

            if image_file:
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)

                if self.label_folder_path:
                    labels = self.get_files(self.label_folder_path, [".txt", ".json", ".xml"])

                    if not labels:
                        st.warning("标签文件夹中没有找到支持的标签格式。请检查路径和标签格式。")
                        return

                    label_file = os.path.splitext(image_file)[0] + ".txt"
                    if label_file in labels:
                        with open(os.path.join(self.label_folder_path, label_file), "r") as file:
                            content = file.read()

                            st.image(image, caption="src", use_column_width=True)
                            st.text_area("标签内容:", value=content, height=200)
                    else:
                        st.warning("未找到对应的标签文件。请确保图像和标签文件具有相同的文件名。")
                else:
                    st.warning("请输入标签文件夹路径。")
        else:
            st.warning("请输入图像文件夹路径。")

    def parse_yolo(self, content):
        # 解析yolo格式的标签文件
        lines = content.strip().split('\n')
        parsed_content = []
        for line in lines:
            values = line.split()
            class_id, x, y, w, h = values
            parsed_content.append((class_id, float(x), float(y), float(w), float(h)))
        return parsed_content

    def parse_xml(self, content):
        from xml.etree import ElementTree as ET
        # 解析xml格式的标签文件
        root = ET.fromstring(content)
        parsed_content = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin, ymin, xmax, ymax = [int(bndbox.find(coord).text) for coord in ['xmin', 'ymin', 'xmax', 'ymax']]
            parsed_content.append((name, xmin, ymin, xmax - xmin, ymax - ymin))
        return parsed_content

    def parse_json(self, content):
        # 解析json格式的标签文件
        data = json.loads(content)
        parsed_content = []
        for item in data:
            class_name = item['class']
            x, y, w, h = item['bbox']
            parsed_content.append((class_name, x, y, w, h))
        return parsed_content

    def draw_bbox(self, image_cv, bboxes):
        class_colors = {}
        for bbox in bboxes:
            class_name, x, y, w, h = bbox
            if class_name not in class_colors:
                # 为每个类别分配一个随机颜色
                class_colors[class_name] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            color = class_colors[class_name]
            cv2.rectangle(image_cv, pt1, pt2, color, 2)
            cv2.putText(image_cv, class_name, (int(x), int(y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        return image_cv

    def visual_detection(self):
        '''
        可视化检测
        :return:
        '''
        if self.image_folder_path:
            images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
            if not images:
                st.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                return

            self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1, value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=self.image_index)

            if image_file:
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)

                if self.label_folder_path:
                    annotations = self.get_files(self.label_folder_path, [".txt", ".json", ".xml"])

                    if not annotations:
                        st.warning("标签文件夹中没有找到支持的标注格式。请检查路径和标注格式。")
                        return

                    label_ext = None
                    for ext in [".txt", ".json", ".xml"]:
                        annotation_file = os.path.splitext(image_file)[0] + ext
                        if annotation_file in annotations:
                            label_ext = ext
                            break

                    if label_ext:
                        with open(os.path.join(self.label_folder_path, annotation_file), "r") as file:
                            content = file.read()

                            if label_ext == ".txt":
                                bboxes = self.parse_yolo(content)
                            elif label_ext == ".xml":
                                bboxes = self.parse_xml(content)
                            elif label_ext == ".json":
                                bboxes = self.parse_json(content)

                            image_cv = cv2.imread(image_path)
                            image_cv = self.draw_bbox(image_cv, bboxes)
                            col1, col2 = st.columns(2)
                            col1.image(image, caption="src", use_column_width=True)
                            col2.image(image_cv, caption="dst", use_column_width=True)
                            st.text_area("标签内容:", value=content, height=200)
                    else:
                        st.warning("未找到对应的标签文件。请确保图像和标签文件具有相同的文件名。")
                else:
                    st.warning("请输入标签文件夹路径。")
        else:
            st.warning("请输入图像文件夹路径。")


    def visual_segmentation(self):
        '''
        可视化分割
        :return:
        '''
        if self.image_folder_path:
            images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
            if not images:
                st.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                return

            self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=self.image_index)

            if image_file:
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)

                if self.label_folder_path:
                    masks = self.get_files(self.label_folder_path, [".png", ".bmp", ".tiff"])
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

                        blend = self.draw_mask(image, mask, self.colors)
                        col1, col2 = st.columns(2)
                        col1.image(image, caption="src", use_column_width=True)
                        col2.image(blend, caption="dst", use_column_width=True)
                    else:
                        st.warning("未找到对应的分割掩码文件。请确保图像和掩码文件具有相同的文件名。")
                else:
                    st.warning("请输入标签文件夹路径。")
        else:
            st.warning("请输入图像文件夹路径。")

    def load_image_preview(self):
        if self.image_folder_path and self.image_index is not None:
            images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
            if images:
                image_file = images[self.image_index]
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)
                st.sidebar.image(image, caption="预览", width=100)
            else:
                st.sidebar.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")

    def visual(self):
        if self.task_type == "分类":
            self.visual_classification()
            self.load_image_preview()
        elif self.task_type == "检测":
            self.visual_detection()
            self.load_image_preview()
        elif self.task_type == "分割":
            self.visual_segmentation()
            self.load_image_preview()
        else:
            st.warning("请选择任务类型")

def main():
    st.title("Dataset.Viewer©️(Hertz)")
    viewer = DatasetViewer()
    viewer.load_sidebar()
    viewer.visual()

if __name__ == "__main__":
    main()


