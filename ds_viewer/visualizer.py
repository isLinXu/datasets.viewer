
import os
import streamlit as st
import cv2
from PIL import Image
import random
import json
import numpy as np

class DatasetViewer:
    def __init__(self):
        self.bboxes = None
        self.mask = None
        self.image_folder_path = ""
        self.label_folder_path = ""
        self.class_names = []
        self.colors = []
        self.image_files = []
        self.label_files = []
        self.task_type = ""
        self.image_index = 0

    def get_files(self, folder_path, supported_formats):
        '''
        获取文件
        :param folder_path:
        :param supported_formats:
        :return:
        '''
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
            if os.path.exists(self.image_folder_path):
                images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
                self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1, value=0)
                # self.confidence_threshold = st.slider("设置置信度阈值:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

                if st.sidebar.button("保存可视化结果"):
                    if self.image_folder_path and self.image_index is not None:
                        images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
                        if images:
                            image_file = images[self.image_index]
                            image_path = os.path.join(self.image_folder_path, image_file)
                            image = Image.open(image_path)
                            self.save_visual_result(image, self.task_type)
                        else:
                            st.sidebar.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                    else:
                        st.sidebar.warning("请输入图像文件夹路径。")

                if st.sidebar.button("转换并导出标签"):
                    src_format = st.sidebar.text_input("输入源标签格式（例如：yolo、xml、json）:", value="")
                    dst_format = st.sidebar.text_input("输入目标标签格式（例如：yolo、xml、json）:", value="")
                    if src_format and dst_format:
                        self.convert_and_export_labels(src_format, dst_format)
                    else:
                        st.sidebar.warning("请输入源标签格式和目标标签格式。")

    def draw_mask(self, image, mask, colors):
        '''
        绘制掩码
        :param image:
        :param mask:
        :param colors:
        :return:
        '''
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

            # self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,value=0)
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

    def parse_yolo(self, content, image_cv):
        '''
        解析yolo格式的标签文件
        :param content:
        :param image_cv:
        :return:
        '''
        # 获取图像的宽度和高度
        img_height, img_width = image_cv.shape[:2]

        # 解析yolo格式的标签文件
        lines = content.strip().split('\n')
        parsed_content = []
        for line in lines:
            class_id, x, y, w, h = map(float, line.strip().split())
            x_min = int((x - w / 2) * img_width)
            x_max = int((x + w / 2) * img_width)
            y_min = int((y - h / 2) * img_height)
            y_max = int((y + h / 2) * img_height)
            class_id_str = str(int(class_id))
            w_ori = x_max - x_min
            h_ori = y_max - y_min
            parsed_content.append((class_id_str, x_min, y_min, w_ori, h_ori))
        return parsed_content

    def parse_label(self, image_file, show_image=True):
        annotation_file = ''
        instance_coco_train_json = ''
        instance_coco_val_json = ''
        image_path = os.path.join(self.image_folder_path, image_file)
        image = Image.open(image_path)
        if self.label_folder_path:
            annotations = self.get_files(self.label_folder_path, [".txt", ".json", ".xml"])

            if not annotations:
                st.warning("标签文件夹中没有找到支持的标注格式。请检查路径和标注格式。")
                return

            label_ext = None
            for ext in [".txt", ".json", ".xml"]:
                if ext == ".json":
                    single_json_file = os.path.splitext(image_file)[0] + ext
                    single_json_path = os.path.join(self.label_folder_path, single_json_file)
                    if os.path.exists(single_json_path):
                        label_ext = ext
                        annotation_file = single_json_file
                        break
                    else:
                        instance_coco_train_json = 'instances_train2017.json'
                        instance_coco_val_json = 'instances_val2017.json'
                        train_coco_json = os.path.join(self.label_folder_path, instance_coco_train_json)
                        val_coco_json = os.path.join(self.label_folder_path, instance_coco_val_json)
                        if os.path.exists(train_coco_json) or os.path.exists(val_coco_json):
                            label_ext = ext
                            annotation_file = instance_coco_train_json if os.path.exists(
                                train_coco_json) else instance_coco_val_json
                            break
                elif ext == ".xml" or ext == ".txt":
                    annotation_file = os.path.splitext(image_file)[0] + ext
                    if annotation_file in annotations:
                        label_ext = ext
                        break

            if label_ext:
                with open(os.path.join(self.label_folder_path, annotation_file), "r") as file:
                    content = file.read()
                    image_cv = cv2.imread(image_path)
                    if label_ext == ".txt":
                        bboxes = self.parse_yolo(content, image_cv)
                    elif label_ext == ".xml":
                        bboxes = self.parse_xml(content)
                    elif label_ext == ".json":
                        if annotation_file in [instance_coco_train_json, instance_coco_val_json]:
                            bboxes = self.parse_json(content, image_file)
                        else:
                            bboxes = self.parse_single_json(content)
                        content = json.dumps(bboxes, indent=4)

                    if show_image:
                        # 保存 bboxes 到实例变量中
                        self.bboxes = bboxes

                        image_cv = self.draw_bbox(image_cv, bboxes)
                        col1, col2 = st.columns(2)
                        col1.image(image, caption="src", use_column_width=True)
                        col2.image(image_cv, caption="dst", use_column_width=True)
                        # st.text_area("标签内容:", value=content, height=200)
                        st.text_area("标签内容:", value=content, height=200,
                                     key=f"label_content_{image_file}")
            else:
                st.warning("未找到对应的标签文件。请确保图像和标签文件具有相同的文件名。")
        else:
            st.warning("请输入标签文件夹路径。")
        return bboxes, content

    def parse_xml(self, content):
        '''
        解析xml格式的标签文件
        :param content:
        :return:
        '''
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

    def parse_single_json(self, content):
        '''
        解析单个图像对应的COCO数据集格式的json标签文件
        :param content:
        :return:
        '''
        # 解析与单个图像对应的COCO数据集格式的json标签文件或自定义格式的json标签文件
        data = json.loads(content)

        if all(key in data for key in ["bbox", "masks", "mask_shape", "scores", "classes"]):
            # 解析自定义格式的json标签文件
            bboxes = data["bbox"]
            class_ids = data["classes"]
            id_to_name = {i: str(i) for i in set(class_ids)}  # 将类别ID映射到类别名称（字符串格式的ID）

            parsed_content = []
            for class_id, bbox in zip(class_ids, bboxes):
                class_name = id_to_name[class_id]
                x1, y1, x2, y2 = bbox
                # 计算宽度和高度
                w = x2 - x1
                h = y2 - y1
                parsed_content.append((class_name, x1, y1, w, h))
        else:
            # 解析COCO数据集格式的json标签文件
            id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

            parsed_content = []
            for ann in data['annotations']:
                class_id = ann['category_id']
                class_name = id_to_name[class_id]  # 从id_to_name字典中获取class_name
                x, y, w, h = ann['bbox']
                parsed_content.append((class_name, x, y, w, h))

        return parsed_content

    def parse_json(self, content, image_file):
        '''
        解析COCO数据集格式的json标签文件
        :param content:
        :param image_file:
        :return:
        '''
        image_file = image_file.split('.')[0]
        print("image_file:", image_file)
        # 解析COCO数据集格式的json标签文件
        data = json.loads(content)
        image_id = None
        for img in data['images']:
            if img['file_name'] == image_file:
                image_id = img['id']
                break

        if image_id is None:
            return []

        # 创建一个字典，将category_id映射到class_name
        id_to_name = {cat['id']: cat['name'] for cat in data['categories']}

        parsed_content = []
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                class_id = ann['category_id']
                class_name = id_to_name[class_id]  # 从id_to_name字典中获取class_name
                x, y, w, h = ann['bbox']
                parsed_content.append((class_name, x, y, w, h))  # 使用class_name而不是class_id
        return parsed_content

    def draw_bbox(self, image_cv, bboxes):
        '''
        绘制边界框
        :param image_cv:
        :param bboxes:
        :return:
        '''
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

            # self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1, value=0)
            image_file = st.sidebar.selectbox("选择图像文件:", images, index=self.image_index)

            if image_file:
                bboxes, content = self.parse_label(image_file)
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

            # self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1,value=0)
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
        '''
        加载图像预览
        :return:
        '''
        if self.image_folder_path and self.image_index is not None:
            images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
            if images:
                image_file = images[self.image_index]
                image_path = os.path.join(self.image_folder_path, image_file)
                image = Image.open(image_path)
                st.sidebar.image(image, caption="预览", width=100)
            else:
                st.sidebar.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")


    def save_visual_result(self, image, task_type):
        global result_image
        save_path = os.path.join(self.image_folder_path, "visual_results")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if self.image_folder_path and self.image_index is not None:
            images = self.get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
            if images:
                image_file = images[self.image_index]
                # "分类", "检测", "分割"
                if task_type == "分类":
                    result_image = image
                elif task_type == "检测":
                    bboxes, content = self.parse_label(image_file, show_image=False)
                    image_path = os.path.join(self.image_folder_path, image_file)
                    image_cv = cv2.imread(image_path)
                    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                    result_image = self.draw_bbox(image_cv, bboxes)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                elif task_type == "分割":
                    # todo: 读取分割掩码
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result_image = self.draw_mask(image_cv, self.mask, self.colors)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                else:
                    st.warning("未知的任务类型。结果图像未被保存。")

            result_image = Image.fromarray(result_image)
            result_image_path = os.path.join(save_path, f"{task_type}_result_{self.image_index}.png")
            result_image.save(result_image_path)
            st.success(f"已保存可视化结果到：{result_image_path}")

    def convert_and_export_labels(self, src_format, dst_format):
        # TODO: 实现不同标签格式之间的转换，并将结果保存到指定的文件夹
        st.warning("标签格式转换功能尚未实现。")

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