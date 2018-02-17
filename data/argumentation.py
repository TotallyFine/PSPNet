# coding:utf-8

import random
from PIL import Image

def random_crop(img, target, size):
    w, h = img.size
    assert w > size and h > size
    if size is not None:
        th, tw = size
        if w == tw and h == th:
            return img, target
        else:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), target.crop((x1, y1, x1 + tw, y1 + th))
    else:
        return img, target
            


def filp(img, target):
    if random.random() < 0.5:
        return img.transpose(Image.FLIP_LEFT_RIGHT), target.transpose(Image.FLIP_LEFT_RIGHT)
    return img, target
