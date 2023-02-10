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
from video_extract import Video_Extractor
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')  # 改变标准输出的默认编码
in_dim = 768
text_extract = Text_Extractor(dim=in_dim)


def inverse(d):
    res = {}
    for item in d:
        res[d[item]] = item
    return res


info = {
        '电影名': 0, '电影网址': 1, '导演': 2, '编剧': 3, '主演': 4,
        '评分': 5, '评分人数': 6, '评分额外信息': 7,
        '类型': 8, '制片国家/地区': 9, '语言': 10, '上映时间': 11, '时长': 12,
        '剧情简介': 13, '获奖情况': 14, '热门评论': 15,
        '海报链接': 16, '海报保存路径': 17, '剧照链接': 18, '剧照保存路径': 19, '视频链接': 20, '视频保存路径': 21
    }
ids = inverse(info)

wjj_list = ['data', 'data1', 'data2', 'data3']
e = 2
head = 79
file_list = os.listdir('./' + wjj_list[e] + '/movie_info')
print(file_list)
t0 = time.time()
nu = 0
for index in range(0, len(file_list)):  # 遍历所有导演的电影信息文件
    t = time.time()
    file = file_list[index]
    path = './' + wjj_list[e] + '/movie_info/' + file
    dir_name = './feat_new_data' + str(e)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    save_path = dir_name + '/' + str(index + head) + '.txt'  # 保存路径

    data = []  # 文件信息列表
    with open(path, 'r', encoding='utf-8') as f:
        csv_read = csv.reader(f)
        for i, row in enumerate(csv_read):
            if i:
                data.append(row)
    num_movie = len(data)  # 导演的电影个数
    # print('num:', num_movie)

    director_index = {}  # 电影索引存储
    # root
    director = data[0][info['导演']]
    print(f'----------{index}-----------')
    print(file)
    print(director)
    with open(save_path, 'a+') as f:
        x = text_extract(director).detach()  # root
        x = torch.squeeze(x)
        l = [k.item() for k in list(x)]
        f.write(str(l) + '\n')
        f.write('num' + '\t' + 'score' + '\t' + 'name' + '\t' + 'info' + '\n')

    k = 0  # 有效电影个数
    print('num_movie=', num_movie)
    for i in range(num_movie):  # 遍历某个导演的所有电影
        d = i
        movie = data[i]
        for j, item in enumerate(movie):
            if item == 'INF':
                movie[j] = '暂无'
            if j < 16:
                movie[j] = re.sub(' ', '', movie[j].strip('\n'))

        if movie[info['剧情简介']] == '暂无':
            continue

        valid = True
        information = []  # 电影各模态信息结点

        # level1
        name = movie[info['电影名']]
        # print('name', name)
        x = text_extract(name).detach()  # root
        x = torch.squeeze(x)
        l = [k.item() for k in list(x)]
        name = str(l)
        # print(name)

        # level2 text-1
        base = []
        actor = '主演是'
        if movie[info['主演']] != '暂无':
            # print(director, name)
            # print(director, name, movie[info['主演']])
            actor += ','.join(eval(movie[info['主演']]))
        else:
            actor += '暂无'
        base.append(actor)
        writer = '编剧是' + re.sub('/', '、', movie[info['编剧']])
        base.append(writer)
        for i in range(8, 13):
            base.append(ids[i] + '是' + re.sub('/', '、', movie[i]))
        base_info = '。'.join(base) + '。'
        # print(base_info)
        x = text_extract(base_info).detach()
        x = torch.squeeze(x)
        l = [k.item() for k in list(x)]
        information.append((len(information), str(l)))

        # level2 text-2
        introduction = 'None'
        if movie[info['剧情简介']] != '暂无':
            introduction = '剧情简介是' + movie[info['剧情简介']]
            x = text_extract(introduction).detach()  # root
            x = torch.squeeze(x)
            l = [k.item() for k in list(x)]
            introduction = str(l)
        else:
            valid = False
        # print(introduction)
        information.append((len(information), introduction))

        # level2 text-3
        # commends = ''.join(eval(movie[info['热门评论']]))
        commends = str([item[:100] for item in eval(movie[info['热门评论']])])
        if not eval(commends):
            commends = 'None'
        # print(commends)
        else:
            x = text_extract(commends).detach()  # root
            x = torch.squeeze(x)
            l = [k.item() for k in list(x)]
            commends = str(l)
        information.append((len(information), commends))

        # level2 text-4
        addition = ''
        if movie[info['评分额外信息']] != '暂无':
            addition += movie[info['评分额外信息']] + '。'
        if movie[info['获奖情况']] != '暂无':
            addition += '。'.join(eval(movie[info['获奖情况']])) + '。'
        if not addition:
            addition = 'None'
        else:
            x = text_extract(addition).detach()  # root
            x = torch.squeeze(x)
            l = [k.item() for k in list(x)]
            addition = str(l)
        # print(addition)
        information.append((len(information), addition))

        # level2 image
        image_path = movie[info['海报保存路径']]
        image = os.listdir(image_path)
        if image:
            image = str([image_path + '/' + item for item in image])
            x = Image_Extractor(image)[2].detach()
            x = torch.squeeze(x)
            l = [k.item() for k in list(x)]
            image = str(l)
        else:
            image = 'None'
        # print(image)
        information.append((len(information), image))

        # level2 video
        video_path = movie[info['视频保存路径']]
        video = os.listdir(video_path)
        if video:
            video = str([video_path + '/' + item for item in video])
            x = Video_Extractor(video).detach()
            x = torch.squeeze(x)
            l = [k.item() for k in list(x)]
            video = str(l)
        else:
            video = 'None'
        # print(video)
        information.append((len(information), video))

        if valid:
            label = movie[info['评分']]
            with open(save_path, 'a+') as f:
                f.write(str(k) + '\t' + label + '\t' + name + '\t' + str(information) + '\n')
                k += 1
            print(str(d)+' ', end='')
    print('')
    director_index[index+head] = (director, k)
    with open('./director_index'+str(e)+'.txt', 'a+') as f:
        f.write(str(director_index) + '\n')
    print('第{}个导演完成！共{}部电影'.format(index+1, k), '  耗时：'+str((time.time()-t)/60)+'分钟。')
    nu += k
    print('已完成{}部电影！'.format(nu))

print('==============================================================')
print('共' + str(nu) + '部电影，整体耗时：'+str((time.time()-t0)/60)+'分钟。')

