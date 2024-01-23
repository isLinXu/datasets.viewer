import json

def parse_yolo(content, image_cv):
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


def parse_xml(content):
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


def parse_single_json(content):
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


def parse_json(content, image_file):
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
