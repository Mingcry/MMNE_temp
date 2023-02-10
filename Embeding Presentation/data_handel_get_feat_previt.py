# -*- coding:utf-8 -*-
import os
import re
import csv
import sys
import io
import time
from text_extract import Text_Extractor
from image_extract import *
from image_extract import Image_Extractor

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码
in_dim = 768

feat_new_data = os.listdir('./feat_new_data')
new_data = os.listdir('./new_data')
new_data.sort(key=lambda x: int(x.split('.')[0]))
feat_new_data.sort(key=lambda x: int(x.split('.')[0]))
save_root = './feat_previt_new_data'
if not os.path.exists(save_root):
    os.mkdir(save_root)
save_paths = [save_root+'/'+i for i in feat_new_data]
feat_new_data = ['./feat_new_data'+'/'+i for i in feat_new_data]
new_data = ['./new_data'+'/'+i for i in new_data]
# print(new_data)
# print(feat_new_data)
# print(save_paths)
al = len(new_data)
nu = 306
t0 = time.time()
for index in range(20, al):  # 遍历所有导演的电影信息文件
    director_index = {}
    t = time.time()
    new_path = new_data[index]
    feat_path = feat_new_data[index]
    save = save_paths[index]

    with open(feat_path, 'r', encoding='utf-8') as f:
        feat_d = f.readlines()
    with open(save, 'a+', encoding='utf-8') as f:
        f.write(feat_d[0])
        f.write(feat_d[1])
    with open(new_path, 'r', encoding='utf-8') as f:
        new_d = [i.strip('\n') for i in f.readlines()]
        director = new_d[0]
    feat_d = [i.strip('\n') for i in feat_d[2:]]
    new_d = new_d[2:]
    for j in range(0, len(feat_d)):
        feat = (feat_d[j]).split('\t')
        new = (new_d[j]).split('\t')
        feat_info = eval(feat[3])

        new_info = eval(new[3])
        img = new_info[4][1]
        if img != 'None':
            x = Image_Extractor(img)[2].detach()
            x = torch.squeeze(x)
            l = [k.item() for k in list(x)]
            image = str(l)

            feat_info[4] = (4, image)
            feat[3] = str(feat_info)

        info = '\t'.join(feat)
        info += '\n'
        with open(save, 'a+', encoding='utf-8') as f:
            f.write(info)

    k = len(feat_d)
    director_index[index] = (director, len(feat_d))
    with open('./director_previt_index.txt', 'a+') as f:
        f.write(str(director_index) + '\n')
    print('第{}个导演完成！共{}部电影'.format(index+1, k), '  耗时：'+str((time.time()-t)/60)+'分钟。')
    nu += k
    print('已完成{}部电影！'.format(nu))
    with open('record_n_previt.txt', 'a+') as ff:
        ff.write(f'index={index+1}  n={nu}\n')


print('==============================================================')
print('共' + str(nu) + '部电影，整体耗时：'+str((time.time()-t0)/60)+'分钟。')

