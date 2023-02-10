import random
import time
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import timm
# from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model
from timm.models.vision_transformer import vit_base_patch16_224 as create_model

# resnet18 = models.resnet18()
# vit = create_model(pretrained=True)
vit = create_model(pretrained=True, num_classes=2)
vit.load_state_dict(torch.load('pre_tune_vit1.pth'))

handle = transforms.Compose([transforms.RandomResizedCrop(224),  # 将PIL图像裁剪成任意大小和纵横比
                             transforms.RandomHorizontalFlip(),  # 以0.5的概率水平翻转给定的PIL图像
                             transforms.RandomVerticalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                            ])
p = 0.8


def Image_Extractor(paths, handle=handle, p=p):
    paths = eval(paths)
    random.shuffle(paths)
    n = max(1, int(p * len(paths)))
    paths = paths[:n]
    pic = []
    for path in paths:
        image = Image.open(path)
        # plt.imshow(image)  # 显示图片
        # plt.show()

        image = handle(image)
        image = torch.unsqueeze(image, 0)
        pic.append(image)
    # print('pic', pic)
    image = torch.cat(tuple(pic), dim=0)
    output = vit(image)
    output_feat = vit.forward_features(image)
    output_mean = torch.mean(output_feat, dim=0)
    output_mean = torch.unsqueeze(output_mean, 0)
    return output, output_feat, output_mean


def demo():
    path = './data/post_video/乌尔善/刀见笑/image/海报3.jpg'  # 540*774图片
    image = Image.open(path).convert('RGB')

    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # image.show()
    image = img_transforms(image)  # H*W*C
    print(image.shape)




if __name__ == '__main__':
    demo()  # 试验
    '''
    paths = "['./data/post_video/乌尔善/刀见笑/image/剧照1.jpg', './data/post_video/乌尔善/刀见笑/image/剧照2.jpg', './data/post_video/乌尔善/刀见笑/image/剧照3.jpg', './data/post_video/乌尔善/刀见笑/image/剧照4.jpg', './data/post_video/乌尔善/刀见笑/image/剧照5.jpg', './data/post_video/乌尔善/刀见笑/image/海报1.jpg', './data/post_video/乌尔善/刀见笑/image/海报2.jpg', './data/post_video/乌尔善/刀见笑/image/海报3.jpg', './data/post_video/乌尔善/刀见笑/image/海报4.jpg', './data/post_video/乌尔善/刀见笑/image/海报5.jpg']"

    output, output_feat, output_mean = Image_Extractor(paths)
    print(output.shape, output_feat.shape, output_mean.shape)
    print(output_mean)
    '''






