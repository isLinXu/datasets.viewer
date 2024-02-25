
from PIL import Image

import sys
sys.path.insert(0, './')
from .draw import *
from .labels import *
from .tools import *

def parse_label(state, st, image_file, show_image=True):
    annotation_file = ''
    instance_coco_train_json = ''
    instance_coco_val_json = ''
    bboxes = None
    content = None
    image_path = os.path.join(state.image_folder_path, image_file)
    image = Image.open(image_path)
    if state.label_folder_path:
        annotations = get_files(state.label_folder_path, [".txt", ".json", ".xml", "box"])

        if not annotations:
            st.warning("标签文件夹中没有找到支持的标注格式。请检查路径和标注格式。")
            return

        label_ext = None
        for ext in [".txt", ".json", ".xml", ".box"]:
            if ext == ".json":
                single_json_file = os.path.splitext(image_file)[0] + ext
                single_json_path = os.path.join(state.label_folder_path, single_json_file)
                if os.path.exists(single_json_path):
                    label_ext = ext
                    annotation_file = single_json_file
                    break
                else:
                    instance_coco_train_json = 'instances_train2017.json'
                    instance_coco_val_json = 'instances_val2017.json'
                    train_coco_json = os.path.join(state.label_folder_path, instance_coco_train_json)
                    val_coco_json = os.path.join(state.label_folder_path, instance_coco_val_json)
                    if os.path.exists(train_coco_json) or os.path.exists(val_coco_json):
                        label_ext = ext
                        annotation_file = instance_coco_train_json if os.path.exists(
                            train_coco_json) else instance_coco_val_json
                        break
            elif ext == ".xml" or ext == ".txt" or ext == ".box":
                annotation_file = os.path.splitext(image_file)[0] + ext
                if annotation_file in annotations:
                    label_ext = ext
                    break

        if label_ext:
            with open(os.path.join(state.label_folder_path, annotation_file), "r") as file:
                content = file.read()
                image_cv = cv2.imread(image_path)
                if label_ext == ".txt":
                    bboxes = parse_yolo(content, image_cv)
                elif label_ext == ".box":
                    bboxes = parse_box(content, image_cv)
                elif label_ext == ".xml":
                    bboxes = parse_xml(content)
                elif label_ext == ".json":
                    if annotation_file in [instance_coco_train_json, instance_coco_val_json]:
                        bboxes = parse_json(content, image_file)
                    else:
                        bboxes = parse_single_json(content)
                    content = json.dumps(bboxes, indent=4)
                state.bboxes = bboxes # 保存 bboxes 到实例变量中
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