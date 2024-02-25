import os
import random
from PIL import Image

from .analyze import analyze_yolo_dataset
from .tools import get_files
from .visual import save_visual_result


def load_sidebar(state, st):
    with st.sidebar:
        st.header("设置")
        state.image_folder_path = st.text_input("输入图像文件夹路径:", value="")
        state.label_folder_path = st.text_input("输入标签文件夹路径:", value="")
        class_names = st.text_input("输入类别名称（用逗号分隔）:", value="")
        if class_names:
            state.class_names = [name.strip() for name in class_names.split(",")]
        num_classes = len(state.class_names) if state.class_names else 100
        state.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)]

        task_options = ["分类", "检测", "分割", "增强"]
        state.task_type = st.selectbox("选择任务类型:", task_options)

        if state.task_type == "检测":
            state.edit_label = st.checkbox("编辑标签")
            if state.edit_label and state.bboxes is not None:
                state.edit_bbox_index = st.number_input("选择要编辑的边界框索引:", min_value=0,
                                                        max_value=len(state.bboxes) - 1, step=1, value=0)
                state.new_class_name = st.text_input("输入新的类别名称（可选）:", value="")
                state.new_bbox = st.text_input("输入新的边界框坐标（格式：xmin,ymin,xmax,ymax，可选）:", value="")
            elif state.edit_label and state.bboxes is None:
                st.warning("没有边界框可供编辑。请确保输入了正确的标签文件夹路径。")

        if os.path.exists(state.image_folder_path):
            images = get_files(state.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
            state.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1,
                                                        step=1, value=state.image_index)
            if st.sidebar.button("分析数据集"):
                dataset_path = os.path.dirname(os.path.dirname(state.label_folder_path))
                classes_to_analyze = None
                class_names = None
                analyze_yolo_dataset(dataset_path, classes_to_analyze, class_names)

            if st.sidebar.button("保存可视化结果"):
                if state.image_folder_path and state.image_index is not None:
                    images = get_files(state.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
                    if images:
                        image_file = images[state.image_index]
                        image_path = os.path.join(state.image_folder_path, image_file)
                        image = Image.open(image_path)
                        save_visual_result(state, st, image)
                    else:
                        st.sidebar.warning("图像文件夹中没有找到支持的图像格式。请检查路径和图像格式。")
                else:
                    st.sidebar.warning("请输入图像文件夹路径。")

            if st.sidebar.button("转换并导出标签"):
                src_format = st.sidebar.text_input("输入源标签格式（例如：yolo、xml、json）:", value="")
                dst_format = st.sidebar.text_input("输入目标标签格式（例如：yolo、xml、json）:", value="")
                if src_format and dst_format:
                    state.convert_and_export_labels(src_format, dst_format)
                else:
                    st.sidebar.warning("请输入源标签格式和目标标签格式。")