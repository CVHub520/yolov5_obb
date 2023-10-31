'''
@author: CVHub
@date: 2023-10-28
'''

import cv2
import numpy as np
import onnxruntime as ort
from random import randint
import argparse

NMS_THRES = 0.2
CONF_THRES = 0.25
PI = 3.141592

# DOTA-v1.5
CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter', 'container-crane'
]

COLORS = [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in CLASSES]


class YOLOv5_OBB:
    def __init__(self, model_path, stride=32):
        self.model_path = model_path
        self.stride = stride

    def poly_label(self, 
                   im, 
                   poly, 
                   label='', 
                   color=(128, 128, 128), 
                   txt_color=(255, 255, 255)):
        lw = max(round(sum(im.shape) / 2 * 0.003), 2)  # line width
        polygon_list = np.array([
            (poly[0], poly[1]), 
            (poly[2], poly[3]),
            (poly[4], poly[5]), 
            (poly[6], poly[7]
        )], np.int32)
        cv2.drawContours(
            image=im, 
            contours=[polygon_list], 
            contourIdx=-1, 
            color=color, 
            thickness=lw
        )
        if label:
            tf = max(lw - 1, 1)  # font thicknes
            xmin, ymin = min(poly[0::2]), min(poly[1::2])
            x_label, y_label = int(xmin), int(ymin)
            w, h = cv2.getTextSize(
                label, 0, fontScale=lw / 3, thickness=tf
            )[0]  # text width, height
            cv2.rectangle(
                            im,
                            (x_label, y_label),
                            (x_label + w + 1, y_label + int(1.5*h)),
                            color, -1, cv2.LINE_AA
                        )
            cv2.putText(
                im, 
                label,
                (x_label, y_label + h), 
                0, 
                lw / 3, 
                txt_color, 
                thickness=tf, 
                lineType=cv2.LINE_AA
            )

    def rbox2poly(self, obboxes):
        """
        Trans rbox format to poly format.
        Args:
            rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

        Returns:
            polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
        """
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)
        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)

    def scale_polys(self, img1_shape, polys, img0_shape, ratio_pad=None):
        # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
        # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = resized / raw
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0] # h_ratios
            pad = ratio_pad[1] # wh_paddings
        polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
        polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
        polys[:, :8] /= gain # Rescale poly shape to img0_shape
        #clip_polys(polys, img0_shape)
        return polys

    def letterbox(self, 
                  im, 
                  new_shape, 
                  color=(114, 114, 114), 
                  auto=False, 
                  scaleFill=False, 
                  scaleup=True):
        """
        Resize and pad image while meeting stride-multiple constraints
        Returns:
            im (array): (height, width, 3)
            ratio (array): [w_ratio, h_ratio]
            (dw, dh) (array): [w_padding h_padding]
        """
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int): # [h_rect, w_rect]
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # wh ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)) # w h 
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0]) # [w h]
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # [w_ratio, h_ratio]

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)

    def preprocess(self, img, new_shape):
        img = self.letterbox(img, new_shape, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype('float32')
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def postprecess(self, prediction, src_img, new_shape):

        nc = prediction.shape[2] - 5 - 180  # number of classes

        xc = prediction[..., 4] > CONF_THRES
        outputs = prediction[:][xc]

        generate_boxes, bboxes, scores = [], [], []

        for out in outputs:

            cx, cy, longside, shortside, obj_score = out[:5]
            class_scores = out[5: 5+nc]
            class_idx = np.argmax(class_scores)

            max_class_score = class_scores[class_idx] * obj_score
            if max_class_score < CONF_THRES:
                continue

            theta_scores = out[5+nc:]
            theta_idx = np.argmax(theta_scores)
            theta_pred = (theta_idx - 90) / 180 * PI

            bboxes.append([[cx, cy], [longside, shortside], max_class_score])
            scores.append(max_class_score)
            generate_boxes.append([
                cx, cy, longside, shortside, 
                theta_pred, max_class_score, class_idx
            ])

        indices = cv2.dnn.NMSBoxesRotated(
            bboxes, scores, CONF_THRES, NMS_THRES
        )
        det = np.array(generate_boxes)[indices.flatten()]

        pred_poly = self.rbox2poly(det[:, :5])

        pred_poly = self.scale_polys(new_shape, pred_poly, src_img.shape)
        det = np.concatenate((pred_poly, det[:, -2:]), axis=1) # (n, [poly conf cls])

        for *poly, conf, cls in reversed(det):
            c = int(cls)
            label = f'{CLASSES[c]} {conf:.2f}'
            self.poly_label(src_img, poly, label, COLORS[c])
        
        cv2.imshow('Result', src_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self, src_img):
        
        net = ort.InferenceSession(
            self.model_path, providers=['CPUExecutionProvider']
        )
        input_name = net.get_inputs()[0].name
        input_shape = net.get_inputs()[0].shape
        new_shape = input_shape[-2:]

        blob = self.preprocess(src_img, new_shape)
        outputs = net.run(None, {input_name: blob})[0]
        self.postprecess(outputs, src_img, new_shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 Object Detection with Oriented Bounding Boxes')
    parser.add_argument('--model', required=True, help='Path to the YOLOv5 model file (ONNX format)')
    parser.add_argument('--image', required=True, help='Path to the input image for detection')
    parser.add_argument('--stride', type=int, default=32, help='Stride value for the model (default: 32)')
    args = parser.parse_args()

    src_img = cv2.imread(args.image)

    yolo = YOLOv5_OBB(model_path=args.model, stride=args.stride)
    yolo.run(src_img)
