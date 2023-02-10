# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import os
from random import *
import subprocess
from torch.autograd import Variable
from Pre_train_model.timesformer.models.transforms import *
from Pre_train_model.timesformer.models.vit import TimeSformer
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_input(image_path, num_frame=16, flag=False):
    prefix = '{:05d}.jpg'
    feat_path = image_path
    images = []

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_params = {
        "side_size": 256,
        "crop_size": 224,
        "num_segments": num_frame,
        "sampling_rate": 5
    }
    transform_val = torchvision.transforms.Compose([
        GroupScale(int(transform_params["side_size"])),
        GroupCenterCrop(transform_params["crop_size"]),
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(mean, std),
    ])
    frame_list = os.listdir(feat_path)
    average_duration = len(frame_list) // transform_params["num_segments"]
    offsets = np.array(
        [int(average_duration / 2.0 + average_duration * x) for x in range(transform_params["num_segments"])])
    offsets = offsets + 1
    if flag:
        print('采样坐标：', offsets)
    for i, seg_ind in enumerate(offsets):
        p = int(seg_ind)
        seg_imgs = Image.open(os.path.join(feat_path, prefix.format(p))).convert('RGB')
        if i == 0 and flag:
            print('*******每帧大小：', seg_imgs.size)
        images.append(seg_imgs)
    video_data = transform_val(images)
    if flag:
        print('帧数量：', len(images))
        print('视频数据初始shape：', video_data.shape)
    video_data = video_data.view((-1, transform_params["num_segments"]) + video_data.size()[1:])
    if flag:
        print("视频数据最终shape：", video_data.shape)
    out = Variable(video_data)

    return out.unsqueeze(0).to(device)


def handle(path, num_frame, flag=False):
    root = 'E:/pytorch/Moudal_tree/Embeding Presentation'
    split_path = path.split('/')
    path = root + path[1:]
    director, movie_name = split_path[3], split_path[4]
    out_image_dir = './video_key_frame/' + director
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)
    out_image_dir += '/' + movie_name
    if not os.path.exists(out_image_dir):
        os.makedirs(out_image_dir)

    # =================视频抽帧======================
    video_name = path.split('/')[-1].split('.')[0]
    out_image_path = os.path.join(out_image_dir, video_name)
    if not os.path.exists(out_image_path):
        os.makedirs(out_image_path)
    if len(os.listdir(out_image_path)) == 0:
        cmd = "ffmpeg -i \"{}\" -vf select='eq(pict_type\,I)' -vsync 2 -s 224*224 -f image2 \"{}/%05d.jpg\"".format(path,
                                                                                                                     out_image_path)
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # =================提取特征======================
    feat = get_input(out_image_path, num_frame=num_frame, flag=flag)
    if flag:
        print('feat:', feat.shape)

    return feat


def extract(model_input):
    # =================模型建立======================
    model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
                        pretrained_model='E:/pytorch/Moudal_tree/timesformer/TimeSformer_divST_8x32_224_K600.pyth')

    model = model.eval().to(device)
    classifer, feature = model(model_input)
    feature = feature.cpu().detach()

    return feature


def Video_Extractor(paths, num_frame=16, flag=False):
    video = []
    path_list = eval(paths)
    shuffle(path_list)
    path_list = path_list[:3]
    for item in path_list:
        video_path = item
        if flag:
            print(video_path)
        feat = handle(video_path, num_frame, flag=flag)
        video.append(feat)
    video = torch.cat(tuple(video), dim=0)
    if flag:
        print('num_video:', len(video))
        print('model_input:', video.shape)
    out = extract(video)
    out_mean = torch.mean(out, dim=0)
    out_mean = torch.unsqueeze(out_mean, 0)
    return out_mean


if __name__ == '__main__':
    num_frame = 16  # 帧数目
    flag = True  # 是否输出中间print

    # 输入形式
    paths = "['./data/post_video/吴宇森/断箭 Broken Arrow/视频1.mp4', './data/post_video/吴宇森/断箭 Broken Arrow/视频2.mp4']"

    out = Video_Extractor(paths, num_frame=num_frame, flag=flag)
    print('===================================')
    print('model_output:', out.shape)

