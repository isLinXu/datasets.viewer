import os

import cv2
import numpy as np
import yaml
from PIL import Image
import sys
import imgaug.augmenters as iaa
from PIL import Image
from streamlit import number_input, button

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


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def visual_data_aug(state, st):
    # 读取aug.yaml文件
    aug_params = read_yaml("aug.yaml")["data_aug"]

    image_file = load_images(state.image_folder_path, state.image_index, state.image_tags)
    if image_file:
        image_path = os.path.join(state.image_folder_path, image_file)
        image = Image.open(image_path)

        # 使用aug.yaml文件中的参数初始化数据增强选项
        rotation_angle = st.number_input("输入旋转角度（0-360）:", min_value=0, max_value=360, step=1,
                                         value=int(aug_params["degrees"] * 360))
        flip_horizontal = st.checkbox("水平翻转", value=aug_params["fliplr"] > 0.5)
        flip_vertical = st.checkbox("垂直翻转", value=aug_params["flipud"] > 0.5)
        scale = st.slider("缩放比例（0.1-2.0）:", min_value=0.1, max_value=2.0, step=0.1, value=aug_params["scale"])
        brightness = st.slider("亮度调整（-0.5-0.5）:", min_value=-0.5, max_value=0.5, step=0.1,
                               value=aug_params["hsv_v"] - 0.5)
        contrast = st.slider("对比度调整（0.5-2.0）:", min_value=0.5, max_value=2.0, step=0.1, value=aug_params["hsv_s"])
        # 添加更多数据增强选项，如噪声、模糊、锐化等
        add_noise = st.checkbox("添加噪声")
        gaussian_blur = st.checkbox("高斯模糊")
        sharpen = st.checkbox("锐化")
        hue_and_saturation = st.checkbox("色调和饱和度调整")
        if st.button("应用数据增强并显示结果"):
            # 创建数据增强序列
            aug_seq = iaa.Sequential([
                iaa.Rotate(rotation_angle),
                iaa.Fliplr(flip_horizontal),
                iaa.Flipud(flip_vertical),
                iaa.ScaleX(scale),
                iaa.ScaleY(scale),
                iaa.Add(brightness * 255),
                iaa.contrast.LinearContrast(alpha=contrast),
                # 添加更多数据增强方法，如噪声、模糊、锐化等
                iaa.GaussianBlur(sigma=3.0) if gaussian_blur else iaa.Noop(),
                iaa.Sharpen(alpha=0.5) if sharpen else iaa.Noop(),
                iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True) if hue_and_saturation else iaa.Noop()
            ])


            # 应用数据增强
            augmented_image = aug_seq(image=np.array(image))


            # 显示原始图像和增强图像
            col1, col2 = st.columns(2)
            col1.image(image, caption="原始图像", use_column_width=True)
            col2.image(augmented_image, caption="增强结果", use_column_width=True)
    else:
        st.warning("请输入图像文件夹路径。")
