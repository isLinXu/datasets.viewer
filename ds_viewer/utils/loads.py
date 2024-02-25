import os
import shutil

from PIL import Image

from .aug import rotate_image
from .tools import get_files


def load_image_preview(state, st):
    '''
    加载图像预览
    :return:
    '''
    if state.image_folder_path and state.image_index is not None:
        images = get_files(state.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
        if images:
            image_file = images[state.image_index]

            image_path = os.path.join(state.image_folder_path, image_file)
            image = Image.open(image_path)
            st.sidebar.image(image, caption="预览", width=100)

            # 添加标记按钮
            current_tag = state.image_tags.get(image_file)
            tag_options = ["bad", "medium", "good"]
            selected_tag = st.sidebar.selectbox("为当前图像选择一个标记:", tag_options,
                                                index=tag_options.index(current_tag) if current_tag else 0)
            if st.sidebar.button("确认标记"):
                state.image_tags[image_file] = selected_tag
                state.save_state()
                st.sidebar.success(f"已将标记 '{selected_tag}' 应用到图像：{image_file}")

            # 批量删除标记为"bad"的图片
            if st.sidebar.button("批量删除标记为'bad'的图片"):
                for image_file, tag in state.image_tags.items():
                    if tag == "bad":
                        image_path = os.path.join(state.image_folder_path, image_file)
                        if os.path.exists(image_path):
                            os.remove(image_path)
                st.sidebar.success(f"已删除所有标记为'bad'的图片,共计{len(state.image_tags)}张图片已删除。")

            # 批量导出标记为"good"的图片
            if st.sidebar.button("批量导出标记为'good'的图片"):
                export_folder = os.path.join(state.image_folder_path, "good_images")
                if not os.path.exists(export_folder):
                    os.makedirs(export_folder)

                for image_file, tag in state.image_tags.items():
                    if tag == "good":
                        src_image_path = os.path.join(state.image_folder_path, image_file)
                        dst_image_path = os.path.join(export_folder, image_file)
                        shutil.copy(src_image_path, dst_image_path)
                st.sidebar.success \
                    (f"已将所有标记为'good'的图片导出到：{export_folder}，共计{len(state.image_tags)}张图片已导出。")
            # 添加删除按钮
            if st.sidebar.button("删除当前图像"):
                os.remove(image_path)  # 删除图像文件
                st.sidebar.success(f"已删除图像：{image_path}")
                state.image_index = state.image_index if state.image_index < len(images) - 1 else state.image_index - 1
                state.save_state()
            # 添加重置按钮
            if st.sidebar.button("重置状态"):
                state.reset_state()
                st.sidebar.success("已重置状态。")

        else:
            st.sidebar.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")