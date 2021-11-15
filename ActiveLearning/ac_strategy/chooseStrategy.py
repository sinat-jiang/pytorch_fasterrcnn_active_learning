# active learning's sampling strategy
import os
from ActiveLearning.tools_and_statistics import file_tools


def uncertain_cal(detections_for_all_images):
    """
    通过不确定度计算来挑选出信息量较大的图片
    Args:
        detections_for_all_images:
    Returns:
        所有图片的不确定度
    """
    allimages_with_uncertainty = []
    allimages_hard_samples = []  # 无法产生检测框的困难样本
    for img_detections in detections_for_all_images:
        # print("-->", img_detections)
        # 计算不确定度
        # 1 - 先找出所有预测框中分类确定性概率最小的，等效于得到所有预测框中不确定性最大的
        if len(img_detections['detections']) == 0:
            # 无法给出检测结果，将这种样本加入困难样本集合，后面按比进行采样
            # print('image', img_detections['img'].split(os.sep)[-1], 'can not be correct detection!')
            allimages_hard_samples.append(img_detections['img'])
            continue
        min_dec = img_detections['detections'][0]
        for detection in img_detections['detections']:
            if detection['score'] < min_dec['score']:
                min_dec = detection
        # 2 - 将所有预测框中最大的不确定度作为整张图像的不确定度，并返回该图像名称和位置
        uncertainty_img = {'uncertainty': 1 - min_dec['score'],
                           'name': min_dec['name'], 'box_points': min_dec['box_points'],
                           'img_path': img_detections['img']}
        allimages_with_uncertainty.append(uncertainty_img)
    # 其中每个元素：{'uncertainty': 67.80550181865692, 'name': 'd',
    #               'img_path': 'E:\\AllDateSets\\MilitaryOB_AC_5_class_2th\\last_images\\images\\0427.jpg',
    #               'box_points': [1015, 356, 1101, 607]}
    return allimages_with_uncertainty, allimages_hard_samples


def sort_list_with_dict_attr(lists):
    """
    按 list 数组中 dict 元素的某个属性（uncertainty）排序，值从大到小
    :return:
    """
    lists_new = []
    length = len(lists)
    for i in range(length):
        max = lists[0]
        for j in range(len(lists)):
            if max['uncertainty'] < lists[j]['uncertainty']:
                max = lists[j]
        lists_new.append(max)
        lists.remove(max)
    return lists_new


def random_sample(imgs_path, xmls_path, M):
    """
    主动学习 - 随机采样策略：根据指定的路径和数目采样数据，返回随机采样结果。
    Args:
        imgs_path: 图片存储路径
        xmls_path: 标注存储路径
        M: 随机挑选数目
    Returns:
        imgs_list：随机采样的图片
        xmls_list：随机采样的标注
    """
    imgs_list = []
    xmls_list = []
    imgs = file_tools.get_all_file_list(imgs_path)
    xmls = file_tools.get_all_file_list(xmls_path)
    if len(imgs) < M:
        print("指定数据量过大，没有足够数据供挑选！")
        return
    # 产生一个有 M 个不重复的随机数的列表
    random_list = file_tools.random_list_generator(0, len(imgs)-1, M)
    # 用随机数抽取文件
    for r in random_list:
        assert imgs[r].split(os.sep)[-1].split('.')[0] == xmls[r].split(os.sep)[-1].split('.')[0]
        imgs_list.append(imgs[r])
        xmls_list.append(xmls[r])
    return imgs_list, xmls_list


def uncertain_sample(detections_for_all_images, M, lambda_=3, sample_log_file=None):
    """
    主动学习 - 不确定度采样策略：根据当前模型返回的所有图片的检测结果，计算图片的不确定度，并按 1：3 采样不确定度靠前的数据和困难样本数据。
    Args:
        detections_for_all_images: 模型产生的检测结果，每张图片返回的检测结果应保持如下形式：
        [
         {'img': './output/00000030.jpg',
          'detections': [
                         {'name': 'bird', 'score': 0.976276159286499, 'box_points': [271, 61, 499, 504]},
                         {'name': 'bird', 'score': 0.9586328864097595, 'box_points': [85, 71, 529, 537]},
                         {'name': 'bird', 'score': 0.4149879515171051, 'box_points': [118, 89, 350, 498]}
                        ]
         },
        ]
        M: 采样数目
        lambda_: 采样比，默认为 3
        :param sample_log_file: 保存每一轮采样结果的 log 文件
    Returns:
        pre_5_percent_images: 采样数据，同时会在当前文件夹下生成一个保存本次采样结果的 txt 文件。

    """
    # 得到所有未标记图片的检测不确定度和困难样本集合
    allimages_with_uncertainty, allimages_hard_samples = uncertain_cal(detections_for_all_images)
    # 按不确定性排序
    sorted_allimages = sort_list_with_dict_attr(allimages_with_uncertainty)
    # 按采样比采样 ------------------------------
    hard_num = int((M / (lambda_ + 1)) * lambda_)
    if hard_num > len(allimages_hard_samples):
        if len(allimages_hard_samples) == 0:
            hard_num = 0
        else:
            # 如果困难样本没有那么多，直接讲所有困难样本采样，剩下的按不确定度排序采样正常样本
            hard_num = len(allimages_hard_samples)
    norm_num = M - hard_num
    # 将 sorted_allimages 转成只包含图片路径的列表形式
    sorted_allimages_names = [name_['img_path'] for name_ in sorted_allimages]
    pre_5_percent_images = allimages_hard_samples[:hard_num] + sorted_allimages_names[:norm_num]
    # 存储本轮采样的图片
    if sample_log_file is not None:
        import time
        txt_file_name = sample_log_file + str(time.time()).split('.')[0] + '.txt'
        file_tools.text_save(txt_file_name, pre_5_percent_images)
    return pre_5_percent_images


if __name__ == '__main__':
    # random1 sampling
    # imgs_list, xmls_list = random_sample('../output/JPEGImages', '../output/Annotations', 3)
    # print(imgs_list)
    # print(xmls_list)

    # uncertain sampling
    from ac import testCustomODModel
    path = os.path.join('../checkpoints', 'model_custom_voc_13_loss_0.10331234335899353.pth')
    detections_for_all_images = testCustomODModel(model_path=path,
                                                  unod_date_path='../output/JPEGImages',
                                                  min_thresh=0.3)
    print(detections_for_all_images)
    pre_5_percent_images = uncertain_sample(detections_for_all_images, 3)
    print(pre_5_percent_images)
