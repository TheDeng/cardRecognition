import cv2
from .model import EastModel
from .helper import restore_rectangle
from .crop import rorate_with_box
import numpy as np
from . import lanms
import torch
from collections import OrderedDict
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def resize_image(im):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_h, resize_w = 512, 512
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def get_detector(model_path):
    input_size = 512
    text_scale = 512
    score_map_thresh = 0.8
    box_thresh = 0.1
    nms_thres = 0.2
    model = EastModel(text_scale, input_size)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model = model.eval()
    #model = torch.nn.DataParallel(model)
    #model.load_state_dict(torch.load(model_path))
    #model = model.cuda()
    #state_dict = torch.load(model_path,map_location='cpu')
    #new_state_dict = OrderedDict()
    #for k, v in state_dict.items():
    #    name = k[7:]  # remove `module.`
    #    new_state_dict[name] = v
    #model.load_state_dict(new_state_dict)
    #model.eval()
    def detector(im):
        
        im = im[:, :, ::-1]
        new_h, new_w, _ = im.shape
        max_h_w_i = np.max([new_h, new_w, input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im_resized, (ratio_h, ratio_w) = resize_image(im_padded)
        tmp = im_resized
        im_resized = im_resized.astype(np.float32)

        # im_resized = transform(image)
        im_resized = im_resized.transpose(2, 0, 1)

        im_resized = torch.from_numpy(im_resized)
        im_resized = im_resized.cuda()
        im_resized = im_resized.unsqueeze(0)
        im_resized = im_resized.cuda()
        F_score, F_geometry = model(im_resized)
        
        F_score = F_score.permute(0, 2, 3, 1)
        F_geometry = F_geometry.permute(0, 2, 3, 1)
        F_score = F_score.data.cpu().numpy()
        F_geometry = F_geometry.data.cpu().numpy()
        
        if len(F_score.shape) == 4:
            F_score = F_score[0, :, :, 0]
            F_geometry = F_geometry[0, :, :, ]
        # filter the score map
        xy_text = np.argwhere(F_score > score_map_thresh)
        # sort the text boxes via the y axis
        xy_text = xy_text[np.argsort(xy_text[:, 0])]
        # restore
        text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4,
                                              F_geometry[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
        boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
        boxes[:, :8] = text_box_restored.reshape((-1, 8))
        boxes[:, 8] = F_score[xy_text[:, 0], xy_text[:, 1]]
        # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
        boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

        # here we filter some low score boxes by the average score map, this is different from the orginal paper
        for i, box in enumerate(boxes):
            mask = np.zeros_like(F_score, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            boxes[i, 8] = cv2.mean(F_score, mask)[0]
        boxes = boxes[boxes[:, 8] > box_thresh]
        if boxes is not None:
            mx = -float('inf')
            index = 0
            for i,box in enumerate(boxes):
                if box[8] > mx:
                    mx = box[8]
                    index = i
            box = sort_poly(boxes[index,:8].reshape((4,2)).astype(np.int32))
            cv2.polylines(tmp, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0),
                          thickness=1)
            box = box.reshape((1,-1)).tolist()[0]
            box = [box[i] / ratio_w if i % 2 == 0 else box[i] / ratio_h for i in range(len(box))]
            imgOut = rorate_with_box(im, box)
            return imgOut,tmp[:,:,::-1]
        return None,None
        # sorted_box = sorted(boxes.tolist(), reverse=True, key=lambda x: x[8])
        # if len(sorted_box) > 0:
        #     box = sorted_box[0]
        #     box = [box[i] / ratio_w if i % 2 == 0 else box[i] / ratio_h for i in range(len(box))]
        #     imgOut = rorate_with_box('', im, box)
        #     return imgOut
    return detector


