# -*- coding:utf-8 -*-
import os
import time
import torch
import pandas as pd
import numpy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from GCN.GCN_init import GCN
from text_extract import Text_Extractor
from image_extract import *
from image_extract import Image_Extractor
# from video_extract import Video_Extractor
import io
import sys
import random
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码

in_dim = 768
out_dim = 400
feat = 'onlyText'
save = 'E:/pytorch/Moudal_tree/Tree_' + feat + '/data/train_data_' + str(out_dim) + '_inited_not_commend_add.txt'
text_extract = Text_Extractor(dim=in_dim)


def create_x_a(num, director, name_movie, text_nodes, image_nodes, video_nodes):
    A = torch.zeros(num, num)
    x = torch.zeros(num, in_dim)
    x[0] = text_extract(director).detach()  # root
    for i in range(1, len(name_movie) + 1):
        A[0][i] = 1
        A[i][0] = 1
        x[i] = text_extract(name_movie[i-1]).detach()
        for j, nod in text_nodes[i-1].items():
            A[i][j] = 1
            A[j][i] = 1
            # print(j, nod)
            x[j] = text_extract(nod).detach()
        if image_nodes[i-1]:
            for u, nod in image_nodes[i-1].items():
                A[i][u] = 1
                A[u][i] = 1
                # print('image', u, nod)
                x[u] = Image_Extractor(nod)[2].detach()
        if video_nodes[i-1]:
            for u, nod in video_nodes[i-1].items():
                A[i][u] = 1
                A[u][i] = 1
                print('video', u, nod)
                # x[u] = Video_Extractor(nod).detach()
    return A, x


def create_x_a_from_file(num, director, name_movie, text_nodes, image_nodes, video_nodes):
    A = torch.zeros(num, num)
    x = torch.zeros(num, in_dim)
    x[0] = torch.tensor(eval(director)).unsqueeze(0)  # root
    for i in range(1, len(name_movie) + 1):
        A[0][i] = 1
        A[i][0] = 1
        x[i] = torch.tensor(eval(name_movie[i-1])).unsqueeze(0)
        for j, nod in text_nodes[i-1].items():
            A[i][j] = 1
            A[j][i] = 1
            # print(j, nod)
            x[j] = torch.tensor(eval(nod)).unsqueeze(0)
    return A, x


wjj = './feat_new_data'
file = os.listdir(wjj)
file.sort(key=lambda e: int(e.split('.')[0]))
t = time.time()
n = 0
if n == 0:
    with open(save, 'a+') as f:
        f.write('number' + '\t' + 'score' + '\t' + 'movie_feature' + '\n')
for r in range(0, len(file)):
    t1 = time.time()
    print('--------r={}-------'.format(r))
    data_path = wjj + '/' + file[r]
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [item.strip('\n') for item in f.readlines()]
    director = data[0]  # 导演
    data = data[2:]  # 原信息
    num_movie = len(data)  # 电影数量
    name_movie = []  # 电影名
    label = []  # 评分
    info = []  # 信息
    for i in range(len(data)):
        movie = data[i].split('\t')[1:]
        label.append(float(movie[0]))
        name_movie.append(movie[1])
        info.append(eval(movie[2]))  # 里面是一个列表

    k = 1 + num_movie  # 存第三层结点
    node_num = 1 + num_movie  # 结点个数
    text_nodes = []
    image_nodes = []
    video_nodes = []
    index = [_ for _ in range(1, 1+num_movie)]
    for item in info:
        text = {}
        image = {}
        video = {}
        for j, node in enumerate(item):
            if node[1] != 'None' and j < 2:  # 文本信息j=0,1,2,3
                text[k] = node[1]
                k += 1
                node_num += 1
        text_nodes.append(text)
        image_nodes.append(image)
        video_nodes.append(video)

    A, x = create_x_a_from_file(node_num, director, name_movie, text_nodes, image_nodes, video_nodes)  # 主要花时间

    model = GCN(A, in_dim, out_dim)
    model.initialize()
    output = model(x).detach()[index]

    for item, y in zip(output, label):
        l = [k.item() for k in list(item)]
        with open(save, 'a+') as f:
            f.write(str(n) + '\t' + str(y) + '\t' + str(l) + '\n')
            n += 1
    print('第{}个导演嵌入完成，共有{}部电影！'.format(r+1, num_movie))
    print('耗时 ' + str((time.time()-t1)/60)[:5] + '分钟')
    print('n={}'.format(n))
    with open('E:/pytorch/Moudal_tree/Tree_' + feat + '/data/record_n_400.txt', 'a+') as ff:
        ff.write(f'n={n}\n')

print('\n=====================================')
print('Finished!')
print('总耗时 ' + str((time.time() - t)/60)[:5] + '分钟')
print('一共嵌入[n={}]部电影！'.format(n))


