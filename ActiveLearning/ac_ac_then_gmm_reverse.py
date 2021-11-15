"""
主动学习（包括训练、富样本选取、更新样本池） + GMM 聚类 两种方式结合来选取数据
相比于 ac_and_train.py 此处采用的方法是：
    先用主动学习的不确定性策略挑选出要标注的数据；
    然后对 tarin 数据进行 gmm 聚类；
    接着将不确定性策略选出的数据输入 gmm 模型，选出离各自聚类中心 较远（即当前模型无法对其有效的聚类） 的数据，加到 train 集中，重新训练；
另外，与 ac_ac-then-gmm.py 区别的是，这里不再是仅训练一个epoch就挑选一次，而是训练 8 次取最好的模型作为一轮得到的模型，
然后用该模型去挑选数据。
"""
import sys
sys.path.append("..")
import datetime
import os
import time
import numpy as np
import cv2
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
from ActiveLearning import filetools, train_data_pro_def
from ActiveLearning.filetools import save_img_detected
from engine import custom_voc_evaluate, train_one_epoch
from train_data_pro_def import get_dataset, get_transform
from GMMClusting import demo


def testCustomODModel(model_path, unod_date_path, min_thresh=0.3):
    """
    使用自己的训练的模型进行目标检测
    Args:
        model_path: 决定使用的模型的参数文件（.pth 文件）存放路径
        unod_date_path: 需要检测的文件所在的文件夹
        min_thresh: bbox 的选择阈值（只保留在该阈值之上的检测框）（由于要计算不确定度，所以设小一点，这样会得到一些不确定度较大的 bbox）
    :return:
    """
    # 加载当前最优的模型（采用重现构建网络结构，然后仅加载网络参数的方式）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(utils.CLASS))
    model.cuda()
    checkpoint = torch.load(f=model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()    # 记得

    # 获取所有未标注的图片，并利用现有模型进行预测
    images = filetools.get_all_file_list(unod_date_path)
    detections_for_all_images = []
    for img in images:
        print(img)
        # 需要将图片转化成 faster-rcnn 模型指定的输入形式
        input = []
        src_img = cv2.imread(img)
        img2 = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        img_tensor = torch.from_numpy(img2 / 255.).permute(2, 0, 1).float().cuda()
        input.append(img_tensor)

        # 检测并返回检测结果
        detection = model(input)
        # print(detection)

        # 从检测结果中分别提取出 bbox、labels 和 scores
        boxes = detection[0]['boxes']
        labels = detection[0]['labels']
        scores = detection[0]['scores']
        assert len(boxes) == len(labels) == len(scores)

        # 保存图片(根据现有模型将大于阈值0.3的框都画到测试图片上去)，若不想将检测结果画出，注释掉即可
        # 先判断文件夹是否存在，若不存在，就重新创建
        if not os.path.exists(os.path.dirname(unod_date_path) + "/images_detected"):
            os.mkdir(os.path.dirname(unod_date_path) + "/images_detected")
        output_img_path = os.path.dirname(unod_date_path) + "/images_detected" + "\\" + img.split('\\')[-1][:-4] + "-detected.jpg"
        print(output_img_path)
        # out_dir, boxes, names, scores, src_img, labels
        save_img_detected(out_dir=output_img_path, boxes=boxes, names=utils.CLASS, scores=scores, src_img=src_img,
                          labels=labels, min_thresh=min_thresh)

        detection_rc = []
        # 选定指定阈值之上的检测结果，并打印出所有的检测结果看看
        # （记得指定检测阈值，不能选的太大，因为后面还要靠此计算不确定度，默认 0.3）
        for idx in range(len(boxes)):
            if scores[idx].item() <= min_thresh:
                continue
            # print(labels[idx], scores[idx], boxes[idx])
            # 重新组织检测结果的结构，以便之后能更方便的解析（对每张图片中的检测结果：
            # 目标类别 ：分数 ：bbox
            # c : 55.584704875946045 : [672, 393, 953, 567]
            # d : 91.02560877799988 : [672, 393, 953, 567]
            # ...
            # ）
            name = utils.CLASS[labels[idx].item()]  # 取出 label 对应的类名
            boxes_np = [int(i.item()) for i in boxes[idx]]
            # print(boxes_np)     # [814.6400146484375, 491.2326354980469, 1081.942138671875, 545.265869140625]
            # score = round(scores[idx].item(), 3)  # round(2.3456, 3) = 2.345  即保留小数点后 3 位
            detection_dict = {'name': name, "score": scores[idx].item(), 'box_points': boxes_np}
            # print('detection_dict:\n', detection_dict)
            detection_rc.append(detection_dict)

        detections_for_all_images.append({'img': img, 'detections': detection_rc})
    return detections_for_all_images


def select_images(detections_for_all_images):
    """
    通过不确定度计算来挑选出信息量较大的图片
    :return: 所有图片的不确定度
    """
    allimages_with_uncertainty = []
    allimages_hard_samples = []  # 无法产生检测框的困难样本
    for img_detections in detections_for_all_images:
        print("-->", img_detections)
        # 计算不确定度
        # 1 - 先找出所有预测框中分类确定性概率最小的，等效于得到所有预测框中不确定性最大的
        if len(img_detections['detections']) == 0:
            # 有可能有的图片无法给出检测结果，此时说明当前模型对该图片无法正常检测，应该是需要该图片来改善模型性能
            # 将这种样本加入困难样本集合，后面按比进行采样
            print('image', img_detections['img'].split('\\')[-1], 'can not be correct detection!')
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


def ac_train(args):
    """为了主动学习模块重定义的训练，主要是每次都要从上次的训练停止处开始训练（加载上一次的参数）"""

    if not args['is_first_train']:  # 不是主动学习的第一次训练时，不要加载预训练模型
        args['pretrained'] = False

    device = torch.device(args['device'])
    # Data loading code
    print("Loading data")
    # 如果是自定义Pascal数据集,不需要传入image_set参数,因此这里设置为None
    dataset, num_classes = get_dataset(get_transform(train=True), args['train_data_path'])
    dataset_test, _ = get_dataset(get_transform(train=False), args['test_data_path'])

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=args['workers'],
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=args['workers'],
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=args['pretrained'])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['lr_steps'], gamma=args['lr_gamma'])

    if not args['is_first_train']:      # 如果不是第一次训练，需要恢复训练
        print('--------------load last model parameters--------------')
        checkpoint = torch.load(args['resume'], map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        # 恢复训练，但是不想使用原来的 动态学习率逻辑，就不要加载原来的 optimizer 和 lr_scheduler
        # optimizer.load_state_dict(checkpoint['optimizer'])  # 用于恢复训练,处理模型还需要优化器和学习率规则
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    # 如果只进行模型测试,注意这里传入的参数是--resume, 原作者只提到了--resume 用于恢复训练,根据官方文档可知也是可以用于模型推理的
    # 参考官方文档https://pytorch.org/tutorials/beginner/saving_loading_models.html
    if args['test_only']:
        if not args['resume']:
            raise Exception('需要checkpoints模型用于推理!')
        else:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            checkpoint = torch.load(args['resume'], map_location='cpu')
            # 加载模型
            model_without_ddp.load_state_dict(checkpoint['model'])

            custom_voc_evaluate(model_without_ddp, data_loader_test, device=device)
            return

    print("Start training")
    start_time = time.time()
    for epoch in range(args['epochs']):
        loss_all = train_one_epoch(model, optimizer, data_loader, device, epoch, args['print_freq'])
        lr_scheduler.step()
        if args['output_dir']:
            # 如果不存在该文件夹，就重新创建
            if not os.path.exists(args['output_dir']):
                os.mkdir(args['output_dir'])
            # model.save('./checkpoints/model_{}_{}.pth'.format(args.dataset, epoch))
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),  # 存储网络参数(不存储网络骨架)
                # 'model': model_without_ddp, # 存储整个网络
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args['output_dir'], 'model_{}_{}_loss_{}.pth'.format('custom_voc', epoch, loss_all)))

            # 保存整个模型和网络参数(取决于后面测试时采用何种方式加载网络)
            # torch.save(model_without_ddp, 'save_name_{}.pkl'.format(epoch))

        custom_voc_evaluate(model, data_loader_test, device=device, thresh=0.8)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def trainer_tmp(model_path, is_train, is_first, test_model_path, valid_path,
                   train_data_path, test_data_path, output_dir):
    """训练"""
    args = {}
    args['device'] = 'cuda'
    args['train_data_path'] = train_data_path
    args['test_data_path'] = test_data_path
    args['workers'] = 4
    # 为了防止 lose 为 nan，记得调整 lr （数据少，就改小）
    args['lr'] = 0.001
    args['momentum'] = 0.9
    args['weight_decay'] = 1e-4
    args['lr_steps'] = [4, 7]
    args['lr_gamma'] = 0.1
    args['epochs'] = 10  # 训练的轮次
    args['print_freq'] = 20
    args['output_dir'] = output_dir  # 模型输出路径
    # 训练
    args['resume'] = None
    args['test_only'] = False
    args['pretrained'] = False  # 是否使用 coco 上预训练的模型

    if not is_train:
        # 测试
        args['pretrained'] = False  # 都是测试了，肯定不会使用 coco 上预训练的模型，而是用自己的模型
        # resume 是保存的网络的各项参数的文件名
        args['resume'] = os.path.join(os.getcwd(), 'checkpoints', test_model_path)
        args['test_only'] = True
        args['test_data_path'] = valid_path
        args['is_first_train'] = False
    else:
        # 若是主动学习的后面的轮次训练，记得不使用预训练模型，但是需要使用上次训练好的参数
        args['is_first_train'] = is_first
        if is_first:
            args['resume'] = None
        else:
            args['resume'] = os.path.join(os.getcwd(), 'checkpoints', model_path)
        # 每次都连续训练 10 次，选取其中 mAP 最高的模型保存
        args['epochs'] = 7  # 不是第一次，之后的迭代训练也要保证所有类的 AP 在指定阈值之上
    ac_train(args)


def ac_and_gmm_choose(model_path, new_images_files, pca_path, M, lambda_):
    """
    两种方式结合挑选数据
    不过这里要求是 M = 72，所以主动学习先排序，然后使用
    """
    # ------------------------ 主动学习先挑选 2M -----------------------------
    # 使用自己训练得到的模型进行目标检测（模型路径、bbox 阈值、不确定度筛选值、选取前百分之几 等参数要改）
    # images_file = r'E:\AllDateSets\MilitaryOB_5_class_torch\unlabel_pool\images'  # 文件路径不能有中文，否则 opencv 会报错
    detections_for_all_images = testCustomODModel(model_path=model_path,
                                                  unod_date_path=new_images_files,
                                                  min_thresh=0.3)  # 还会将高于 min_thresh 的框都画到图像中去（images_detected 文件夹）
    print(detections_for_all_images)
    # 得到所有未标记图片的检测不确定度和困难样本集合
    allimages_with_uncertainty, allimages_hard_samples = select_images(detections_for_all_images)
    print('-------------------- uncertainty_for_allimages ---------------------')
    for unc in allimages_with_uncertainty:
        print(unc)
    print("images number:", len(allimages_with_uncertainty))
    print('--------------------------------------------------------------------')
    print('--------------------------- hard samples ---------------------------')
    for unc in allimages_hard_samples:
        print(unc)
    print("images number:", len(allimages_hard_samples))
    print('--------------------------------------------------------------------')

    # # 挑选出不确定度较大的，即富含更多信息的图片（不确定度阈值暂定为 0.4，挑选比例暂定为前 40%）
    # # 先选出不确定度阈值在 0.4 以上的(一般选出来的都会大于 40，所以这一步移除的图片数量很少)
    # for uncertainty in allimages_with_uncertainty:
    #     if uncertainty['uncertainty'] < 0.4:
    #         # 将小于 50% 的移除
    #         allimages_with_uncertainty.remove(uncertainty)
    # print("after deleting :", len(allimages_with_uncertainty))

    # 按不确定性排序
    sorted_allimages = sort_list_with_dict_attr(allimages_with_uncertainty)
    # 按采样比采样 ------------------------------
    hard_num = int((2*M / (lambda_ + 1)) * lambda_)
    if hard_num > len(allimages_hard_samples):
        if len(allimages_hard_samples) == 0:
            hard_num = 0
        else:
            # 如果困难样本没有那么多，直接讲所有困难样本采样，剩下的按不确定度排序采样正常样本
            hard_num = len(allimages_hard_samples)
    norm_num = 2 * M - hard_num
    # 需要将 sorted_allimages 转成只包含图片路径的列表形式
    sorted_allimages_names = [name_['img_path'] for name_ in sorted_allimages]
    pre_5_percent_images = allimages_hard_samples[:hard_num] + sorted_allimages_names[:norm_num]
    print("--------------------pre_5_percent_images---------------------")
    for per in pre_5_percent_images:
        print(per)
    print("--------------------------------------------------------------")

    # ------------------------ GMM 在此基础上再挑选 M（这里是对训练集进行聚类） -----------------------------
    # 1 - 进行 pca 并保存参数
    # pca_path = r'E:\AllDateSets\MilitaryOB_5_class_torch_3th\train\JPEGImages'
    # 注意：在外部文件调用含 matplotlib 的函数一般是画不出来数据分布的
    # 可能的报错：ValueError: setting an array element with a sequence.此时说明图片的大小可能不一致，需调整
    # 这里就简单的调整为第一张图片的大小
    m, n = Image.open(pre_5_percent_images[0]).size
    demo.pca_opt(pca_path, m, n, if_show=False, pickle_model='military_pca_modes_ac_then_gmm.pkl')

    # 2 - GMM Clusting and choose those img un-near the kernal of class
    # data_path = pca_path
    pickle_model = 'military_pca_modes_ac_then_gmm.pkl'  # PCA 参数模型
    K = 5  # 聚类数
    main_comp = 25  # 主成分数（默认为25，后面会在函数内动态调整）
    if_show = False  # 是否将聚类结果画出来
    # near_ = 0.25     # 选取靠近聚类中心的数据比例
    test_data_list = filetools.get_all_file_list(pca_path)
    if len(pre_5_percent_images) > len(test_data_list):
        main_comp = int(len(test_data_list) * 0.9)
    else:
        main_comp = int(len(pre_5_percent_images) * 0.9)  # 主成分数
    nearest_data, nearest_data_near = demo.GMMClusting_opt_3th(pre_5_percent_images, pickle_model, M, K,
                                                              test_data_list, main_comp, if_show)
    # for i in range(len(sorted_data)):
    #     for k in sorted_data[i]:
    #         print(k.name, k.score, k.label)
    #     print(len(nearest_data[i]))
    # print('----------------------------------')
    # 调整 list（只保存挑选出来的图像名）
    # 逆序选择远离聚类中心的数据
    unnearst_data_near = nearest_data[-M:]
    unnearest_data_list = []
    for k in unnearst_data_near:
        print(k.name, k.score, k.label)
        unnearest_data_list.append(k.name)
    print('after GMM choose, there has:', len(unnearest_data_list), 'images!')

    # 将选出的图片数据存入文件
    import time
    txt_file = 'pre_5_percent_images_ac_then_gmm_' + str(time.time()).split('.')[0] + '.txt'
    filetools.text_save(txt_file, unnearest_data_list)
    return txt_file


if __name__ == '__main__':
    # 训练 或 测试
    # train_data_path = r"/home/jiangb/dataset/for_ac_then_gmm_reverse_choose/train"   # 训练集
    # test_data_path = r"/home/jiangb/dataset/for_ac_then_gmm_reverse_choose/test"     # 测试集
    # model_path = 'ac_then_gmm_reverse_choose/5/model_custom_voc_2_loss_0.4986625015735626.pth'     # 当前轮使用的初始模型
    # output_dir = './checkpoints/ac_then_gmm_reverse_choose/6'  # 模型输出路径
    # test_model_path = 'xxx'    # 用于测试的模型路径
    # valid_path = r'xxx'  # 验证集的路径（路径不能包含中文）
    # is_train = True     # 是训练还是测试
    # is_first = False    # 是否是初始训练轮次
    # trainer_tmp(model_path, is_train, is_first, test_model_path, valid_path,
    #                train_data_path, test_data_path, output_dir)    # 记得去 函数内去改参数

    # 先用不确定性策略挑选出 2M 张图片（这其中包含 hard sample），再用 GMM 聚类选出远离聚类中心的前 M 张图片
    M = 72
    model_name = 'model_custom_voc_6_loss_0.024218659847974777.pth'
    model_path = os.path.join(os.getcwd(), 'checkpoints', 'ac_then_gmm_reverse_choose', '6', model_name)
    choose_file = r'/home/jiangb/dataset/for_ac_then_gmm_reverse_choose/unlabelpool/images'
    # pca_path 是测试数据集的路径（因为要选择与测试集数据分布靠近的数据）
    pca_path = r'/home/jiangb/dataset/for_ac_then_gmm_reverse_choose/test/JPEGImages'
    h_u_lambda = 3     # 困难样本 ：不确定度采样样本 = 1 ：lambda_
    txt_file_name = ac_and_gmm_choose(model_path, choose_file, pca_path, M, lambda_=h_u_lambda)
    # 转移文件(同时转移图片和标注文件)
    current_path = os.getcwd()
    # file_path = os.path.join(current_path, 'pre_10_percent_images_2th_1588504100.txt')
    file_path = os.path.join(current_path, txt_file_name)
    to_file_path = r"/home/jiangb/dataset/for_ac_then_gmm_reverse_choose/unlabelpool/挑选文件临时展示"
    # 将 txt 文件中包含的图片文件及其对应的标注文件都转移到 test 文件夹下（防止出错），方便手动将其转到 train 文件夹中
    filetools.remove_imgs(file_path, to_file_path=to_file_path, file_struct='str')
    
    # 将图片移到 train 集中
    
