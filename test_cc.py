# coding:utf-8
from __future__ import print_function
import cfgs_360cc as cfgs
import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils import *
import argparse


# ---------------------network
def load_network():
    model_fe = cfgs.net_cfgs['FE'](**cfgs.net_cfgs['FE_args'])

    cfgs.net_cfgs['CAM_args']['scales'] = model_fe.Iwantshapes()
    # print(cfgs.net_cfgs['CAM_args']['scales'])
    model_cam = cfgs.net_cfgs['CAM'](**cfgs.net_cfgs['CAM_args'])

    model_dtd = cfgs.net_cfgs['DTD'](**cfgs.net_cfgs['DTD_args'])

    if cfgs.net_cfgs['init_state_dict_fe'] != None:
        model_fe.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_fe'], map_location=torch.device('cpu')))
    if cfgs.net_cfgs['init_state_dict_cam'] != None:
        model_cam.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_cam'], map_location=torch.device('cpu')))
    if cfgs.net_cfgs['init_state_dict_dtd'] != None:
        model_dtd.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_dtd'], map_location=torch.device('cpu')))

    # model_fe.cuda()
    # model_cam.cuda()
    # model_dtd.cuda()
    return (model_fe, model_cam, model_dtd)

# def parse_arg():
#     parser = argparse.ArgumentParser(description="demo")
#
#     parser.add_argument('--image_path', type=str, default='images/test.png', help='the path to your image')
#     parser.add_argument('--checkpoint', type=str, default='models/360CC/exp2E0_I32000-136650_M',
#                         help='the path to your checkpoints')
#
#     args = parser.parse_args()


    return args
# ---------------------testing stage
def recognition(img, model, decoder, device):

    inp_h = 192
    inp_w = 2048
    mean = np.array(0.588, dtype=np.float32)
    std = np.array(0.193, dtype=np.float32)
    # ratio resize
    img_h, img_w = img.shape
    img = cv2.resize(img, (0, 0), fx=inp_w / img_w, fy=inp_h / img_h, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (inp_h, inp_w, 1)) # int(img_w*inp_h/img_h)
    img = img.astype(np.float32)
    img = (img / 255. - mean) / std
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)
    img = img.to(device)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)

    model[0].eval()
    model[1].eval()
    model[2].eval()
    features = model[0](img)
    A = model[1](features)
    output, out_length = model[2](features[-1], A, torch.tensor([1.,2,3]).long(), 10, True)
    sim_pred = decoder.decode(output)
    print('results: {0}'.format(sim_pred))

class decoder():
    def __init__(self, dict_file):
        self.dict = []
        lines = open(dict_file, 'r', encoding='UTF-8').readlines()
        for line in lines:
            self.dict.append(line.replace('\n', ''))
    def decode(self, net_out):
    # decoding prediction into text with geometric-mean probability
    # the probability is used to select the more realiable prediction when using bi-directional decoders
        net_out = F.softmax(net_out, dim = 1).topk(1)[1][:,0].tolist()
        current_text = ''.join([self.dict[_-1] if _ > 0 and _ <= len(self.dict) else '' for _ in net_out])
        return current_text
# ---------------------------------------------------------
# --------------------------Begin--------------------------
# ---------------------------------------------------------
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model = load_network()
print('preparing done')

# args = parse_arg()


started = time.time()

img = cv2.imread('images/test_8.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

decode = decoder('dict/char_std_5990.txt')

recognition(img, model, decode, device)

finished = time.time()
print('elapsed time: {0}'.format(finished - started))