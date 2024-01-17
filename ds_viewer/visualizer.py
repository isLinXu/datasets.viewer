import os
import streamlit as st
import cv2
from PIL import Image
import random


class DatasetViewer:
    def __init__(self):
        self.image_folder_path = ""
        self.label_folder_path = ""
        self.class_names = []
        self.colors = []
        self.image_files = []
        self.label_files = []

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

    def visual(self):
        if self.image_folder_path:
            image_files = os.listdir(self.image_folder_path)
            # 添加更多图像文件格式支持
            supported_formats = [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]
            images = [file for file in image_files if any(file.endswith(fmt) for fmt in supported_formats)]

            # 选择图像文件
            image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,
                                                  value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=image_index)

            if image_file:
                # 显示图像
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)

                if self.label_folder_path:
                    label_files = os.listdir(self.label_folder_path)
                    # 添加更多标注文件格式支持
                    supported_formats = [".txt", ".json", ".xml"]
                    annotations = [file for file in label_files if any(file.endswith(fmt) for fmt in supported_formats)]

                    # 显示标注
                    annotation_file = os.path.splitext(image_file)[0] + ".txt"
                    if annotation_file in annotations:
                        with open(os.path.join(self.label_folder_path, annotation_file), "r") as file:
                            content = file.read()
                            # 在图像上绘制边界框
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
                                    cv2.putText(image_cv, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,0.5, color,2)
                            # 将图像从BGR颜色空间转换为RGB颜色空间
                            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                            col1, col2 = st.columns(2)
                            col1.image(image, caption="src", use_column_width=True)
                            col2.image(image_cv, caption="dst", use_column_width=True)

                            st.write("标注内容:")
                            st.write(content)
                    else:
                        st.write("未找到对应的标注文件")


def main():
    st.title("Dataset.Viewer©️(Hertz)")
    viewer = DatasetViewer()
    viewer.load_sidebar()
    viewer.visual()


if __name__ == "__main__":
    main()
