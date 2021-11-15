import numpy as np
import random
import os
import shutil


def get_all_file_list(root_path):
    """
    返回当前目录下的所有文件列表(注意此函数要求文件名为数字,因为此函数返回按文件名排序后的文件列表)
    :param root_path: 当前目录
    :param key: 指定的排序逻辑需截取的文件名的哪一部分的 lambda 函数
    :return:
    """
    file_list = []
    # key=lambda x: int(x[:-4]) : 倒着数第四位'.'为分界线，按照'.'左边的数字从小到大排序
    for f in sorted(os.listdir(os.path.join(root_path)), key=lambda x: int(x[:-4])):   # 获取的列表是乱序的，记得排一下序
        sub_path = os.path.join(root_path, f)
        if os.path.isfile(sub_path):
            file_list.append(sub_path)
    return file_list


def random_list_generator(start, stop, length):
    """
    返回不包含重复数字的随机数列表，产生的随机数字在 [start, stop) 之间，其个数为 length
    :param start: 起始值（可能包含在产生的列表内）
    :param stop: 终止值（不会包含在产生的列表内）
    :param length: 产生的列表的长度
    :return:
    """
    if length > stop - start + 1:
        print("长度不符合要求")
        return
    random_list = []
    count = 0
    while True:
        # randint(low, high):返回随机的整数，位于半开区间[low, high)
        random_num = random.randint(start, stop)
        if not random_list.__contains__(random_num):
            random_list.append(random_num)
            count += 1
        if count >= length:
            break
    return random_list


def random_divide_initial_trainset(imgs_path, xmls_path, to_img_file_path, to_xml_file_path, percent=0.1, m=None):
    # 随机选取一批数据作为主动学习初始训练数据(默认初始为 10% 的 unlabelpool 数据)
    if not os.path.exists(to_xml_file_path):
        os.mkdir(to_xml_file_path)
    if not os.path.exists(to_img_file_path):
        os.mkdir(to_img_file_path)
    # 首先利用随机数构建图片和标注文件名称
    imgs_list = []
    xmls_list = []
    imgs = get_all_file_list(imgs_path)
    xmls = get_all_file_list(xmls_path)
    if m is None:
        M = int(percent * len(imgs))
    else:
        M = m
    assert len(imgs) == len(xmls)
    if len(imgs) < M:
        print("指定数据量过大，没有足够数据供挑选！")
    # 产生一个有 M 个不重复的随机数的列表
    random_list = random_list_generator(0, len(imgs) - 1, M)
    # 用随机数抽取文件
    for r in random_list:
        # print(r)
        print(imgs[r].split(os.sep)[-1].split('.')[0], xmls[r].split(os.sep)[-1].split('.')[0])
        assert imgs[r].split(os.sep)[-1].split('.')[0] == xmls[r].split(os.sep)[-1].split('.')[0]
        imgs_list.append(imgs[r])
        xmls_list.append(xmls[r])
    print(imgs_list)
    print(xmls_list)
    # 转移文件
    for img, xml in zip(imgs_list, xmls_list):
        print(img, xml)
        shutil.move(img, os.path.join(to_img_file_path, img.split(os.sep)[-1]))
        shutil.move(xml, os.path.join(to_xml_file_path, xml.split(os.sep)[-1]))
    print(os.getcwd())



if __name__ == '__main__':
    # 随机划分主动学习的初始训练集（10%）
    imgs_path = r'unlabelpool/JPEGImages'
    xmls_path = r'unlabelpool/Annotations'
    to_img_file_path = r'train/JPEGImages'
    to_xml_file_path = r'train/Annotations'
    percent = 0.1
    M = None
    random_divide_initial_trainset(imgs_path, xmls_path, to_img_file_path, to_xml_file_path, percent=percent, m=M)


