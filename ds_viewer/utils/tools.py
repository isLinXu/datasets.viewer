import os
import streamlit as st

def get_files(folder_path, supported_formats):
    '''
    获取文件
    :param folder_path:
    :param supported_formats:
    :return:
    '''
    files = os.listdir(folder_path)
    return [file for file in files if any(file.endswith(fmt) for fmt in supported_formats)]


def load_images(image_folder_path, image_index, image_tags):
    if os.path.exists(image_folder_path):
        images = get_files(image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
        if images:
            total_images = len(images)
            progress_bar = st.progress(0)
            for i, image_file in enumerate(images):
                progress = (i + 1) / total_images
                progress_bar.progress(progress)
                if i == image_index:
                    return image_file
        else:
            st.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
    else:
        st.warning("请输入图像文件夹路径。")