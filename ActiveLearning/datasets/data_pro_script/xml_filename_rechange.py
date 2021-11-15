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


def text_save(filename, datas):
    """
    将数据写入 txt 文件
    :param filename: 为写入CSV文件的路径
    :param data: 为要写入数据列表
    :return:
    """
    file = open(filename, 'w')  # 参数 w 表示写（重新写），a 表示追加；
    for data in datas:
        file.write(data.__str__() + '\n')
    file.close()
    print("------保存文件成功------")


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


def random_list_generator(start, stop, length):
    """
    返回不包含重复数字的随机数列表，产生的随机数字在 [start, stop] 之间，其个数为 length
    :param start: 起始值（可能包含在产生的列表内）
    :param stop: 终止值（可能包含在产生的列表内）
    :param length: 产生的列表的长度
    :return:
    """
    if length > stop - start + 1:
        print("长度不符合要求")
        return
    random_list = []
    count = 0
    while True:
        # randint(low, high):返回随机的整数，位于毕区间[low, high]
        random_num = random.randint(start, stop)
        if not random_list.__contains__(random_num):
            random_list.append(random_num)
            count += 1
        if count >= length:
            break
    return random_list


def choosemodel(model_id, model_story_path):
    # 根据当前轮次选出要使用的模型
    # 1.读取当前文件目录下的所有模型文件(按文件名称排序)
    models = []
    for f in sorted(os.listdir(os.path.join(model_story_path))):  # 获取的列表是乱序的，记得排一下序
        sub_path = os.path.join(model_story_path, f)
        if os.path.isfile(sub_path):
            models.append(sub_path)
    return models[model_id].split(os.sep)[-1]


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


def check_label():
    # 统计数据集标签中各类别目标实例的个数
    # 首先读取所有的 xml 文件
    xml_path = r'/home/jiangb/D2_acl_tl/cityspace_alt/faster-rcnn-baseline-target/test/Annotations'
    xml_files = get_all_file_list(xml_path)
    print(len(xml_files))
    classes = {}
    for xml in xml_files:
        doc = read_xml(xml)
        root = doc.getroot()
        obj = root.findall('object')
        for ob in obj:
            c = ob.find('name').text
            if c in classes:
                classes[c] += 1
            else:
                classes[c] = 1
    for key, val in classes.items():
        print(key, val)


def change_xml_filename_with_file(file_path):
    """
    将 xml 中的 filename 修改成和文件名一致
    """
    # 首先得到所有的文件（list）
    files = []
    for f in sorted(os.listdir(os.path.join(file_path)), key=lambda x: int(x[:-4])):  # 获取的列表是乱序的，记得排一下序
        sub_path = os.path.join(file_path, f)
        if os.path.isfile(sub_path):
            files.append(sub_path)
    print('---------------------- 看看是否按序排列了 ------------------------')
    for ik in files:
        print(ik)
    print("lens:", len(files))
    print('----------------------------------------------------------------')
    for file in files:
        doc = read_xml(file)
        root = doc.getroot()
        sub1 = root.find('filename')
        # 修改标签内容
        sub1.text = file.split(os.sep)[-1].split('.')[0] + '.jpg'
        # 保存修改
        doc.write(file)  # 保存修改


if __name__ == '__main__':	
    # 修改 xml filename
    file_name = 'test/Annotations'
    file_name = 'train/Annotations'
    change_xml_filename_with_file(file_name)
