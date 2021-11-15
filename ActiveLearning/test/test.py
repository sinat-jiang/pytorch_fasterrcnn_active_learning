import os

import cv2
# import torch
import xml.etree.ElementTree as ET

import torch

from tools_and_statistics import file_tools

# img = r'/home/jiangb/D2_acl_tl/ATL/cityspace_atl/only-ac-baseline-target/uncertainsampling/unlabelpool/JPEGImages/00004846.jpg'
#
# src_img = cv2.imread(img)
# img2 = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
# print(img2.shape)
# img_tensor = torch.from_numpy(img2 / 255.).permute(2, 0, 1).float().cuda()
# print(img_tensor.shape)
# print(img_tensor)


def read_xml(in_path):
    """
    读取并解析 xml 文件
    :param in_path:
    :return:
    """
    tree = ET.parse(in_path)
    return tree



limit_dict={"person":1,"car":1,"motorcycle":1}
def cout_class_info_with_limit(path, limit_dict):
    from_files = os.listdir(path)
    from_files.sort()
    l=[]
    for file in from_files:
        flag=False
        doc = read_xml(os.path.join(path, file))
        root = doc.getroot()
        objs = root.findall('object')
        for obj in objs:
            c = obj.find('name').text
            if c in limit_dict:
                l.append(file)
                flag=True
                break
        if not flag:
            print(file)
    print(len(l))


if __name__ == '__main__':
    # file_ = r'E:\ac_pro_code_editions\2021-4-16\pytorch_fasterrcnn_detection\ActiveLearning\testdata\source_aachen_000000_000019_leftImg8bit.xml'
    # class_count_dict = {}
    # doc = read_xml(file_)
    # root = doc.getroot()
    # sub1 = root.find('filename')
    # obj = root.findall('object')
    # for ob in obj:
    #     c = ob.find('name').text
    #     if c in class_count_dict.keys():
    #         class_count_dict[c] += 1
    #     else:
    #         class_count_dict[c] = 1
    # print(len(obj))
    # print(class_count_dict)

    # 按记录移动文件

    # train = r'/home/jiang/ATL/jiang/pytorch_fasterrcnn_detection/ActiveLearning/datasets/sim10k/train'
    # test_file = r'/ActiveLearning/datasets/sim10k/test.txt'
    #
    # lines = file_tools.read_txt_file(test_file)
    # print(lines)
    #
    # imgs = []
    # xmls = []
    # for line in lines:
    #     img = os.path.join(train, 'JPEGImages', line + '.jpg')
    #     xml = os.path.join(train, 'Annotations', line + '.xml')
    #     imgs.append(img)
    #     xmls.append(xml)
    # print(imgs)
    # print(xmls)
    # print(len(imgs), len(xmls))
    # to_file = r'/ActiveLearning/datasets/sim10k/test'
    # file_tools.remove_img_and_xml_file(imgs, xmls, to_file)

    # t = torch.tensor(1.9)
    # print(t)
    # print(t.item())
    # print(int(t.item()))

    path = '/home/jiang/ATL/jiang/pytorch_fasterrcnn_detection/ActiveLearning/datasets/cityscape/for_al/unlabelpool/Annotations'
    cout_class_info_with_limit(path, limit_dict)


