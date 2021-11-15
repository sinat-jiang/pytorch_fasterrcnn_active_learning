import os
import xml.etree.ElementTree as ET
import shutil
import random


def read_xml(in_path):
    """
    读取并解析 xml 文件
    :param in_path:
    :return:
    """
    tree = ET.parse(in_path)
    return tree


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


def write_txt(filename, datas, type='w'):
    """
    将数据写入 txt 文件
    :param filename: 为写入CSV文件的路径
    :param data: 为要写入数据列表
    :param type: 参数 w 表示写（重新写），a 表示追加；
    :return:
    """
    file = open(filename, type)
    for data in datas:
        file.write(data.__str__() + '\n')
    file.close()
    print("------保存文件成功------")


def read_txt_file(file_path):
    """
    按行读取 txt 文件内容，返回一个包含每一行内容的 list
    Args:
        file_path: txt 文件路径
    Returns: 返回 list（存储每一行内容）
    """
    lists = []
    f = open(file_path, 'r')  # 读文件
    for line_ in f:
        # print(line_.rstrip())
        lists.append(line_.rstrip())  # 需要使用 rstrip() 取出行末的换行符
    f.close()
    return lists


def remove_img_and_xml_file(imgs, xmls, to_dir_path):
    """
    将 imgs 和 xmls 转移到指定的文件目录下：
    xml 对应 Annotations 文件夹，imgs 对应 JPEGImages 文件夹
    传入的路径均为绝对路径
    """
    # 先判断目标文件夹是否存在，不存在就重建
    imgs_dir = os.path.join(to_dir_path, 'JPEGImages')
    xmls_dir = os.path.join(to_dir_path, 'Annotations')
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir)
    if not os.path.exists(xmls_dir):
        os.makedirs(xmls_dir)

    for img in imgs:
        # 转移图片文件
        shutil.move(img, imgs_dir)
    print("----img 转移成功----")

    for xml in xmls:
        shutil.move(xml, xmls_dir)
    print("----xml 转移成功----")



if __name__ == '__main__':
    # 按记录移动文件

    train = r'./train'
    test_file = r'./test.txt'

    lines = read_txt_file(test_file)
    print(lines)

    imgs = []
    xmls = []
    for line in lines:
        img = os.path.join(train, 'JPEGImages', line + '.jpg')
        xml = os.path.join(train, 'Annotations', line + '.xml')
        imgs.append(img)
        xmls.append(xml)
    print(imgs)
    print(xmls)
    print(len(imgs), len(xmls))
    to_file = os.path.join(os.path.dirname(train), 'test')
    print(to_file)
    remove_img_and_xml_file(imgs, xmls, to_file)

    # 随机划分主动学习的初始训练集（10%）
    # imgs_path = r'/home/jiang/ATL/jiang/pytorch_fasterrcnn_detection/ActiveLearning/datasets/sim10k/for_al/unlabelpool/JPEGImages'
    # xmls_path = r'/home/jiang/ATL/jiang/pytorch_fasterrcnn_detection/ActiveLearning/datasets/sim10k/for_al/unlabelpool/Annotations'
    # to_img_file_path = r'/home/jiang/ATL/jiang/pytorch_fasterrcnn_detection/ActiveLearning/datasets/sim10k/for_al/train/JPEGImages'
    # to_xml_file_path = r'/home/jiang/ATL/jiang/pytorch_fasterrcnn_detection/ActiveLearning/datasets/sim10k/for_al/train/Annotations'
    # percent = 0.1
    # M = None
    # random_divide_initial_trainset(imgs_path, xmls_path, to_img_file_path, to_xml_file_path, percent=percent, m=M)
