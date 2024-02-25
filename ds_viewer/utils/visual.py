import os

import cv2
import numpy as np
from PIL import Image
import sys

from streamlit import number_input, button

from .aug import rotate_image
# sys.path.insert(0, './')
from .draw import draw_bbox, draw_mask
from .parse import parse_label
from .tools import get_files, load_images

def save_visual_result(state, st, image):
    task_type = state.task_type
    global result_image
    save_path = os.path.join(state.image_folder_path, "visual_results")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if state.image_folder_path and state.image_index is not None:
        images = get_files(state.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
        if images:
            image_file = images[state.image_index]
            # "分类", "检测", "分割"
            if task_type == "分类":
                result_image = image
            elif task_type == "检测":
                bboxes, content = parse_label(state, st, image_file, show_image=False)
                image_path = os.path.join(state.image_folder_path, image_file)
                image_cv = cv2.imread(image_path)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
                result_image = draw_bbox(image_cv, bboxes)
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            elif task_type == "分割":
                # todo: 读取分割掩码
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                result_image = draw_mask(image_cv, state.mask, state.colors)
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            else:
                st.warning("未知的任务类型。结果图像未被保存。")

        result_image = Image.fromarray(result_image)
        result_image_path = os.path.join(save_path, f"{task_type}_result_{state.image_index}.png")
        result_image.save(result_image_path)
        st.success(f"已保存可视化结果到：{result_image_path}")


def visual_detection(state,st):
    '''
    可视化检测
    :return:
    '''
    image_file = load_images(state.image_folder_path, state.image_index, state.image_tags)
    if image_file:
        parse_label(state, st, image_file, show_image=True)



def visual_segmentation(state, st):
    '''
    可视化分割
    :return:
    '''
    image_file = load_images(state.image_folder_path, state.image_index, state.image_tags)
    if image_file:
        image_path = os.path.join(state.image_folder_path, image_file)
        image = Image.open(image_path)

        if state.label_folder_path:
            masks = get_files(state.label_folder_path, [".png", ".bmp", ".tiff"])
            if not masks:
                st.warning("标签文件夹中没有找到支持的分割掩码格式。请检查路径和掩码格式。")
                return

            mask_file = os.path.splitext(image_file)[0] + ".png"
            if mask_file in masks:
                mask_path = os.path.join(state.label_folder_path, mask_file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is None:
                    st.warning("无法读取分割掩码。请检查掩码文件。")
                    return

                if mask.shape[0] != image.height or mask.shape[1] != image.width:
                    st.warning("图像和分割掩码的尺寸不匹配。请确保它们具有相同的尺寸。")
                    return

                blend = draw_mask(image, mask, state.colors)
                col1, col2 = st.columns(2)
                col1.image(image, caption="src", use_column_width=True)
                col2.image(blend, caption="dst", use_column_width=True)
            else:
                st.warning("未找到对应的分割掩码文件。请确保图像和掩码文件具有相同的文件名。")
        else:
            st.warning("请输入标签文件夹路径。")
    else:
        st.warning("请输入图像文件夹路径。")


def visual_classification(state, st):
    '''
    可视化分类
    :return:
    '''
    image_file = load_images(state.image_folder_path, state.image_index, state.image_tags)
    if image_file:
        image_path = os.path.join(state.image_folder_path, image_file)
        image = Image.open(image_path)

        if state.label_folder_path:
            labels = get_files(state.label_folder_path, [".txt"])

            if not labels:
                st.warning("标签文件夹中没有找到支持的标签格式。请检查路径和标签格式。")
                return

            label_file = os.path.splitext(image_file)[0] + ".txt"
            if label_file in labels:
                with open(os.path.join(state.label_folder_path, label_file), "r") as file:
                    content = file.read()

                    st.image(image, caption="src", use_column_width=True)
                    st.text_area("标签内容:", value=content, height=200)
            else:
                st.warning("未找到对应的标签文件。请确保图像和标签文件具有相同的文件名。")
        else:
            st.warning("请输入标签文件夹路径。")
    else:
        st.warning("请输入图像文件夹路径。")


def visual_data_aug(state, st):
    image_file = load_images(state.image_folder_path, state.image_index, state.image_tags)
    rotation_angle = number_input("输入旋转角度（0-360）:", min_value=0, max_value=360, step=1, value=0)
    if image_file:
        image_path = os.path.join(state.image_folder_path, image_file)
        image = Image.open(image_path)
        col1, col2 = st.columns(2)
        col1.image(image, caption="src", use_column_width=True)
        if button("应用旋转增强并显示结果"):
            rotated_image = rotate_image(image, rotation_angle)
            col2.image(rotated_image, caption="旋转增强结果", use_column_width=True)
