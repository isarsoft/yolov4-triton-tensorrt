from boundingbox import BoundingBox

import cv2
import numpy as np

def preprocess(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.
    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference
    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            new_h = int(img_h * new_w / img_w)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            new_w = int(img_w * new_h / img_h)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.
    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
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

    keep = np.array(keep)
    return keep

def postprocess(output, img_w, img_h, input_shape, conf_th=0.8, nms_threshold=0.5, letter_box=False):
    """Postprocess TensorRT outputs.
    # Args
        output: list of detections with schema [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold
        letter_box: boolean, referring to _preprocess_yolo()
    # Returns
        list of bounding boxes with all detections above threshold and after nms, see class BoundingBox
    """
    # filter low-conf detections
    detections = output.reshape((-1, 7))
    detections = detections[detections[:, 4] * detections[:, 6] >= conf_th]

    if len(detections) == 0:
        boxes = np.zeros((0, 4), dtype=np.int)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)
    else:
        box_scores = detections[:, 4] * detections[:, 6]

        # scale x, y, w, h from [0, 1] to pixel values
        old_h, old_w = img_h, img_w
        offset_h, offset_w = 0, 0
        if letter_box:
            if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
                old_h = int(input_shape[0] * img_w / input_shape[1])
                offset_h = (old_h - img_h) // 2
            else:
                old_w = int(input_shape[1] * img_h / input_shape[0])
                offset_w = (old_w - img_w) // 2
        detections[:, 0:4] *= np.array(
            [old_w, old_h, old_w, old_h], dtype=np.float32)

        # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = _nms_boxes(cls_detections, nms_threshold)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0)

        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        if letter_box:
            xx = xx - offset_w
            yy = yy - offset_h
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        boxes = boxes.astype(np.int)
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5].astype(np.int)
    detected_objects = []
    for box, score, label in zip(boxes, scores, classes):
        detected_objects.append(BoundingBox(label, score, box[0], box[2], box[1], box[3], img_w, img_h))
    return detected_objects
