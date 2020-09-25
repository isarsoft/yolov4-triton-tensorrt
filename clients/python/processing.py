from boundingbox import BoundingBox

import cv2
import numpy as np

INPUT_HEIGHT = 608
INPUT_WIDTH = 608

def preprocess(image):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(np.array(image, dtype=np.float32, order='C'), (2, 0, 1))
    image /= 255.0
    return image

def nms(boxes, box_confidences, nms_threshold=0.5):
        x_coord = boxes[:, 0]
        y_coord = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        areas = width * height
        ordered = box_confidences.argsort()[::-1]

        keep = list()
        while ordered.size > 0:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
            yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
            xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
            yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)

            iou = intersection / union

            indexes = np.where(iou <= nms_threshold)[0]
            ordered = ordered[indexes + 1]
        keep = np.array(keep).astype(int)
        return keep

def postprocess(buffer, image_width, image_height, conf_threshold=0.8, nms_threshold=0.5):
    detected_objects = []
    img_scale = [image_width / INPUT_WIDTH, image_height / INPUT_HEIGHT, image_width / INPUT_WIDTH, image_height / INPUT_HEIGHT]
    num_bboxes = int(buffer[0, 0, 0, 0])

    if num_bboxes:
        bboxes = buffer[0, 1 : (num_bboxes * 7 + 1), 0, 0].reshape(-1, 7)
        labels = set(bboxes[:, 5].astype(int))
        for label in labels:
            selected_bboxes = bboxes[np.where((bboxes[:, 5] == label) & ((bboxes[:, 4] * bboxes[:, 6]) >= conf_threshold))]
            selected_bboxes_keep = selected_bboxes[nms(selected_bboxes[:, :4], selected_bboxes[:, 4] * selected_bboxes[:, 6], nms_threshold)]
            for idx in range(selected_bboxes_keep.shape[0]):
                box_xy = selected_bboxes_keep[idx, :2]
                box_wh = selected_bboxes_keep[idx, 2:4]
                score = selected_bboxes[idx, 4] * selected_bboxes[idx, 6]

                box_x1y1 = box_xy - (box_wh / 2)
                box_x2y2 = np.minimum(box_xy + (box_wh / 2), [INPUT_WIDTH, INPUT_HEIGHT])
                box = np.concatenate([box_x1y1, box_x2y2])
                box *= img_scale

                if box[0] == box[2]:
                    continue
                if box[1] == box[3]:
                    continue
                detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], image_height, image_width))
    return detected_objects