import random
import cv2
import numpy as np


def draw_mask(image, mask, colors):
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


def draw_bbox(image_cv, bboxes):
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
