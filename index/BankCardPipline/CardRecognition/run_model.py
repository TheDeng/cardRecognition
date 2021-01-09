import os
import time
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
from nltk.metrics.distance import edit_distance

from .utils import CTCLabelConverter, AttnLabelConverter, Averager
from .dataset import hierarchical_dataset, AlignCollate,ResizeNormalize
from .model import Model
from .enhance import equal_hist_with_opt
from PIL import Image
import torchvision.transforms as transforms

def get_res_from_img(model_path):
    opt = dict()
    opt['batch_size'] = 1
    opt['saved_model'] = model_path
    opt['batch_max_length'] = 25
    opt['character'] = '0123456789' # 0123456789|0123456789_
    opt['input_channel'] = 1 # the number of input channel of Feature extractor
    opt['output_channel'] = 512 # the number of output channel of Feature extractor
    opt['hidden_size'] = 256 # the size of the LSTM hidden state

    max_length = opt['batch_max_length']
    converter = CTCLabelConverter(opt['character'])
    opt['num_class'] = len(converter.character)

    model = Model(opt)
    #model = torch.nn.DataParallel(model).cuda()
    # load model
    if opt['saved_model'] != '':
        state_dict = torch.load(opt['saved_model'],map_location='cpu')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    """ evaluation """
    model.eval()

    def recognitioner(image):
        #toW = max(100,int(1.2 * image.size[0] * (32 * 1.0 / image.size[1] * 1.0)))
        toW = 400
        transform = ResizeNormalize((toW, 32))
        toTensor = transforms.ToTensor()    
        image = transform(image)
        image = image.unsqueeze(0)
        batch_size = image.size(0)
        
        with torch.no_grad():
            length_for_pred = torch.cuda.IntTensor([max_length] * batch_size)
            text_for_pred = torch.cuda.LongTensor(batch_size, max_length + 1).fill_(0)

        preds = model(image, text_for_pred).log_softmax(2)
        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2)  # to use CTCloss format
        # Select max probabilty (greedy decoding) then decode index to character
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data)

        # calculate accuracy.
        return sim_preds[0]
    return recognitioner

if __name__ == '__main__':
    

    img_path = 'card.png'
    model_path = "saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth"
    recognizer = get_res_from_img(model_path)
    
    # for imgs
    # img_list = os.listdir(img_path)
    # correct = 0
    # total = 0
    # for p in img_list:
    #     image = Image.fromarray(equal_hist_with_opt(path=img_path + p))
    #     pred = recognizer(image)
    #     if '(' in p:
    #         ret = p.split('(')[0].strip()
    #     else:
    #         ret = p.split('.')[0]

    #     if ret == pred.replace('_', ''):
    #         correct += 1
    #     else:
    #         print(p)
    #         print(pred.replace('_', '') + '\n')
    #     total += 1
    # print('total: {0}, correct: {1}'.format(total, correct))
    #
    # for single img

    image = Image.fromarray(equal_hist_with_opt(path=img_path))
    pred = recognizer(image)
    print(pred)