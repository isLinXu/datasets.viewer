
from PIL import Image
import sys

sys.path.insert(0, '/utils/')
from utils.draw import *
from utils.labels import *
from utils.tools import *

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

            if self.task_type == "检测":
                self.edit_label = st.checkbox("编辑标签")
                if self.edit_label and self.bboxes is not None:
                    self.edit_bbox_index = st.number_input("选择要编辑的边界框索引:", min_value=0,
                                                           max_value=len(self.bboxes) - 1, step=1, value=0)
                    self.new_class_name = st.text_input("输入新的类别名称（可选）:", value="")
                    self.new_bbox = st.text_input("输入新的边界框坐标（格式：xmin,ymin,xmax,ymax，可选）:", value="")
                elif self.edit_label and self.bboxes is None:
                    st.warning("没有边界框可供编辑。请确保输入了正确的标签文件夹路径。")

            if os.path.exists(self.image_folder_path):
                images = get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
                self.image_index = st.sidebar.number_input("选择图像文件索引:", min_value=0, max_value=len(images) - 1, step=1, value=0)
                # self.confidence_threshold = st.slider("设置置信度阈值:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

                if st.sidebar.button("保存可视化结果"):
                    if self.image_folder_path and self.image_index is not None:
                        images = get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
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

    def edit_labels(self):
        if self.edit_label:
            if self.new_class_name:
                self.bboxes[self.edit_bbox_index]['class_name'] = self.new_class_name

            if self.new_bbox:
                xmin, ymin, xmax, ymax = [int(coord.strip()) for coord in self.new_bbox.split(",")]
                self.bboxes[self.edit_bbox_index]['bbox'] = (xmin, ymin, xmax, ymax)

    def visual_classification(self):
        '''
        可视化分类
        :return:
        '''
        image_file = load_images(self.image_folder_path, self.image_index)
        if image_file:
            image_path = os.path.join(self.image_folder_path, image_file)
            image = Image.open(image_path)

            if self.label_folder_path:
                labels = get_files(self.label_folder_path, [".txt"])

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

    def parse_label(self, image_file, show_image=True):
        annotation_file = ''
        instance_coco_train_json = ''
        instance_coco_val_json = ''
        bboxes = None
        content = None
        image_path = os.path.join(self.image_folder_path, image_file)
        image = Image.open(image_path)
        if self.label_folder_path:
            annotations = get_files(self.label_folder_path, [".txt", ".json", ".xml"])

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
                        bboxes = parse_yolo(content, image_cv)
                    elif label_ext == ".xml":
                        bboxes = parse_xml(content)
                    elif label_ext == ".json":
                        if annotation_file in [instance_coco_train_json, instance_coco_val_json]:
                            bboxes = parse_json(content, image_file)
                        else:
                            bboxes = parse_single_json(content)
                        content = json.dumps(bboxes, indent=4)
                    self.bboxes = bboxes # 保存 bboxes 到实例变量中
                    if show_image:
                        image_cv = draw_bbox(image_cv, bboxes)
                        col1, col2 = st.columns(2)
                        col1.image(image, caption="src", use_column_width=True)
                        col2.image(image_cv, caption="dst", use_column_width=True)
                        st.text_area("标签内容:", value=content, height=200,
                                     key=f"label_content_{image_file}")
            else:
                st.warning("未找到对应的标签文件。请确保图像和标签文件具有相同的文件名。")
        else:
            st.warning("请输入标签文件夹路径。")
        return bboxes, content

    def visual_detection(self):
        '''
        可视化检测
        :return:
        '''
        image_file = load_images(self.image_folder_path,self.image_index)
        if image_file:
            self.parse_label(image_file)

    def visual_segmentation(self):
        '''
        可视化分割
        :return:
        '''
        image_file = load_images(self.image_folder_path, self.image_index)
        if image_file:
            image_path = os.path.join(self.image_folder_path, image_file)
            image = Image.open(image_path)

            if self.label_folder_path:
                masks = get_files(self.label_folder_path, [".png", ".bmp", ".tiff"])
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

                    blend = draw_mask(image, mask, self.colors)
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
            images = get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
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
            images = get_files(self.image_folder_path, [".jpg", ".png", ".jpeg", ".bmp", ".tiff"])
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
                    result_image = draw_bbox(image_cv, bboxes)
                    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                elif task_type == "分割":
                    # todo: 读取分割掩码
                    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    result_image = draw_mask(image_cv, self.mask, self.colors)
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