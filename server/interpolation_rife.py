import sys
from pathlib import Path

# haxx for RIFE
PY_ROOT = Path(__file__).parent
RIFE_ROOT = PY_ROOT / "arXiv2020_RIFE"
sys.path.insert(0, str(RIFE_ROOT))


import os
import torch
import argparse
from torch.nn import functional as F

from arXiv2020_RIFE.model import RIFE_HDv2

rife_model = RIFE_HDv2.Model()
rife_model.load_model(RIFE_ROOT / 'train_log', -1)
rife_model.eval()
rife_model.device()

# if args.img[0].endswith('.exr') and args.img[1].endswith('.exr'):
#     img0 = cv2.imread(args.img[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
#     img1 = cv2.imread(args.img[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
#     img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device)).unsqueeze(0)
#     img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device)).unsqueeze(0)

# else:
#     img0 = cv2.imread(args.img[0])
#     img1 = cv2.imread(args.img[1])
#     img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
#     img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

@torch.no_grad()
def rife_infer(img0, img1):
    n, c, h, w = img0.shape
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)
    ratio = None
    rthreshold=None
    exp=2

    # hack - disable ratio
    if ratio:
        img_list = [img0]
        img0_ratio = 0.0
        img1_ratio = 1.0
        if ratio <= img0_ratio + args.rthreshold / 2:
            middle = img0
        elif ratio >= img1_ratio - args.rthreshold / 2:
            middle = img1
        else:
            tmp_img0 = img0
            tmp_img1 = img1
            for inference_cycle in range(args.rmaxcycles):
                middle = rife_model.inference(tmp_img0, tmp_img1)
                middle_ratio = ( img0_ratio + img1_ratio ) / 2
                if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                    break
                if args.ratio > middle_ratio:
                    tmp_img0 = middle
                    img0_ratio = middle_ratio
                else:
                    tmp_img1 = middle
                    img1_ratio = middle_ratio
        img_list.append(middle)
        img_list.append(img1)
    else:
        img_list = [img0, img1]
        for i in range(exp):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = rife_model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp
            
    return img_list
