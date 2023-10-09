import json
from torchvision.utils import draw_bounding_boxes
import os
import glob
import torch

def get_bound_box_labels(file_path="valid_labels.json"):
    with open(file_path) as f:
        data = json.load(f)
        label = [x['label'][0] for x in data]    
    label_start = label[::2]
    label_end = label[1::2]

    return label_start, label_end

def draw_bbox(img, label):
    ### get image size
    img = (img*255).type(torch.uint8)
    w, h = img.shape[2], img.shape[1]
    x_mult = w/100
    y_mult = h/100
    x0, y0, x1, y1 = label["x"]*x_mult, label["y"]*y_mult, (label["x"]+label["width"])*x_mult, (label["y"]+label["height"])*y_mult

    img = draw_bounding_boxes(img, torch.tensor([[x0, y0, x1, y1]]), width=1, colors='red')
    return img.type(torch.float32) / 255

def get_paths_from_dir(dir_path):
    paths = glob.glob(os.path.join(dir_path, 'im*.jpg'))
    try:
        paths = sorted(paths, key=lambda x: int((x.split('/')[-1].split('.')[0])[3:]))
    except:
        print(paths)
    return paths
