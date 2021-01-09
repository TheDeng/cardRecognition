from helper import restore_rectangle
from model import EastModel
import cv2
import torch
import numpy as np
import lanms
import os
from crop import rorate_with_box

import warnings  
warnings.filterwarnings("ignore")

def resize_image(im, max_side_len=2400):
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

def test(model, im, filename, input_size = 512, use_cuda = False, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    im = im[:, :, ::-1]
    im_resized, (ratio_h, ratio_w) = resize_image(im)
    tmp = im_resized
    im_resized = im_resized.astype(np.float32)
    im_resized = im_resized.transpose(2, 0, 1)

    im_resized = torch.from_numpy(im_resized)
    im_resized = im_resized.cuda()
    im_resized = im_resized.unsqueeze(0)
    im = im_resized
    if use_cuda:
        im = im.cuda()
    model = model.eval()
    F_score, F_geometry = model(im)
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
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, F_geometry[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = F_score[xy_text[:, 0], xy_text[:, 1]]
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)

    if boxes.shape[0] == 0:
        return [None]*3

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
    
    return boxes,ratio_h, ratio_w,tmp[:,:,::-1]

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    model_path = 'models/model_202.pkl'
    input_size = 512
    text_scale = 512
    model = EastModel(text_scale, input_size)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path))
    if use_cuda:
        model = model.cuda()
    img_root = 'test_images'
    for f in os.listdir(img_root):
        img_path = os.path.join(img_root, f)
        im = cv2.imread(img_path)
        if im is None:
            continue
        boxes,ratio_h, ratio_w,tmp = test(model, im, f, use_cuda=use_cuda, input_size=input_size)
        if boxes is None:
            continue
        print(boxes)
        sorted_box = sorted(boxes.tolist(), reverse=True, key=lambda x: x[8])
        if len(sorted_box) > 0:
            box = sorted_box[0]
            box = [box[i] / ratio_w if i % 2 == 0 else box[i] / ratio_h for i in range(len(box))]
            imgOut = rorate_with_box(im, box)
            cv2.imwrite('card_num_imgs/%s'%f,imgOut)
            cv2.imwrite('labeled_imgs/%s'%f,tmp)
    
