"""
主动学习（包括训练、富样本选取、更新样本池）
采用随机挑选的策略在每轮挑选一批数据加入训练集。
"""
import datetime
import shutil
import time
import sys
import os
sys.path.append("..")
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils
from engine import custom_voc_evaluate, train_one_epoch
from train_data_pro_def import get_dataset, get_transform
from ActiveLearning.tools_and_statistics import file_tools
from ActiveLearning.ac_strategy import chooseStrategy


def ac_train(args):
    """为了主动学习模块重定义的训练，主要是每次都要从上次的训练停止处开始训练（加载上一次的参数）"""

    if not args['is_first_train']:  # 不是主动学习的第一次训练时，不要加载预训练模型
        args['pretrained'] = False

    device = torch.device(args['device'])
    # Data loading code
    print("Loading data")
    # 如果是自定义Pascal数据集,不需要传入image_set参数,因此这里设置为None
    dataset, num_classes = get_dataset(get_transform(train=True), args['train_data_path'], CLASS=args['CLASS'])
    dataset_test, _ = get_dataset(get_transform(train=False), args['test_data_path'], CLASS=args['CLASS'])

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
        # 不使用上次结束时定义的 lr
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
    max_mAP = 0
    max_mAP_idx = 0
    retStrList = []
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

        new_mAP, retStr = custom_voc_evaluate(model, data_loader_test, device=device, CLASS=args['CLASS'], thresh=0.8)
        retStrList.append(retStr)
        if new_mAP > max_mAP:
            max_mAP = new_mAP
            max_mAP_idx = epoch
            
    if utils.LOG_FILEPATH:
        with open(utils.LOG_FILEPATH, 'a') as f:
            f.write(str(max_mAP_idx))
            f.write('\n')
            f.write(retStrList[max_mAP_idx])
            f.write('\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return max_mAP, max_mAP_idx


def trainer_tmp(model_path, is_train, is_first, test_model_path, valid_path,
                   train_data_path, test_data_path, output_dir, CLASS):
    """训练"""
    args = {}
    args['CLASS'] = CLASS
    args['device'] = 'cuda'
    args['train_data_path'] = train_data_path
    args['test_data_path'] = test_data_path
    args['workers'] = 4
    # 为了防止 lose 为 nan，记得调整 lr （数据少，就改小）
    args['lr'] = 0.001
    args['momentum'] = 0.9
    args['weight_decay'] = 1e-4
    args['lr_steps'] = [4, 7]   # 前4轮以 0.001 的学习率训练，后3轮以 0.0001 的学习率训练
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
            args['resume'] = os.path.join(os.getcwd(), model_path)
        # 每次都连续训练 10 次，选取其中 mAP 最高的模型保存
        args['epochs'] = 1  # 不是第一次，之后的迭代训练也要保证所有类的 AP 在指定阈值之上
    max_mAP, max_mAP_idx = ac_train(args)
    return max_mAP, max_mAP_idx


if __name__ == '__main__':
    dataset = 'tmp'
    ac_type = 'random1'

    max_mAP = 0
    max_mAP_idx = 0

    utils.IF_CREATE_LOG_FILT = True
    utils.LOG_FILEPATH = None

    CLASS = None
    train_data_path = None
    test_data_path = None
    output_dir_root = None

    M = 0
    imgs_file_path = None
    xmls_file_path = None
    txt_file_name_root = None
    to_img_file_path = None
    to_xml_file_path = None

    if dataset == 'sim10k':
        utils.LOG_FILEPATH = r'./checkpoints/active_learning/' + dataset + '/ac_log_' + ac_type + '.txt'

        CLASS = ["__background__", 'person', 'car', 'motorbike']
        train_data_path = r'./datasets/' + dataset + '/active_learning/train'  # 训练集
        test_data_path = r'./datasets/' + dataset + '/active_learning/test'  # 测试集
        output_dir_root = r'./checkpoints/active_learning/' + dataset + '/' + ac_type + '/'

        M = 350
        imgs_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/JPEGImages'
        xmls_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/Annotations'
        txt_file_name_root = './each_ite_choose_file_record/' + dataset + '/' + ac_type + '/baseline_' + ac_type + '_5_percent_on_target_'
        to_img_file_path = r'./datasets/' + dataset + '/active_learning/train/JPEGImages'
        to_xml_file_path = r'./datasets/' + dataset + '/active_learning/train/Annotations'
    elif dataset == 'cityscape':
        utils.LOG_FILEPATH = r'./checkpoints/active_learning/' + dataset + '/ac_log_' + ac_type + '.txt'

        CLASS = ["__background__", 'person', 'car', 'motorcycle']
        train_data_path = r'./datasets/' + dataset + '/active_learning/train'  # 训练集
        test_data_path = r'./datasets/' + dataset + '/active_learning/test'  # 测试集
        output_dir_root = r'./checkpoints/active_learning/' + dataset + '/' + ac_type + '/'

        M = 103
        imgs_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/JPEGImages'
        xmls_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/Annotations'
        txt_file_name_root = './each_ite_choose_file_record/' + dataset + '/' + ac_type + '/baseline_' + ac_type + '_5_percent_on_target_'
        to_img_file_path = r'./datasets/' + dataset + '/active_learning/train/JPEGImages'
        to_xml_file_path = r'./datasets/' + dataset + '/active_learning/train/Annotations'
    elif dataset == 'cityscape_single_obj':
        utils.LOG_FILEPATH = r'./checkpoints/active_learning/' + dataset + '/ac_log_' + ac_type + '.txt'

        CLASS = ["__background__", 'car']
        train_data_path = r'./datasets/' + dataset + '/active_learning/train'  # 训练集
        test_data_path = r'./datasets/' + dataset + '/active_learning/test'  # 测试集
        output_dir_root = r'./checkpoints/active_learning/' + dataset + '/' + ac_type + '/'

        M = 99
        imgs_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/JPEGImages'
        xmls_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/Annotations'
        txt_file_name_root = './each_ite_choose_file_record/' + dataset + '/' + ac_type + '/baseline_' + ac_type + '_5_percent_on_target_'
        to_img_file_path = r'./datasets/' + dataset + '/active_learning/train/JPEGImages'
        to_xml_file_path = r'./datasets/' + dataset + '/active_learning/train/Annotations'
    elif dataset == 'cityscape_cityscapefoggy':
        utils.LOG_FILEPATH = r'./checkpoints/active_learning/' + dataset + '/ac_log_' + ac_type + '.txt'
        CLASS = ['__background__', 'car', 'bicycle', 'person', 'bus', 'motorcycle', 'rider', 'truck', 'train']
        train_data_path = r'./datasets/' + dataset + '/active_learning/train'  # 训练集
        test_data_path = r'./datasets/' + dataset + '/active_learning/test'  # 测试集
        output_dir_root = r'./checkpoints/active_learning/' + dataset + '/' + ac_type + '/'

        M = 121
        imgs_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/JPEGImages'
        xmls_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/Annotations'
        txt_file_name_root = './each_ite_choose_file_record/' + dataset + '/' + ac_type + '/baseline_' + ac_type + '_5_percent_on_target_'
        if not os.path.exists('./each_ite_choose_file_record/' + dataset + '/' + ac_type):
            os.makedirs('./each_ite_choose_file_record/' + dataset + '/' + ac_type)
        to_img_file_path = r'./datasets/' + dataset + '/active_learning/train/JPEGImages'
        to_xml_file_path = r'./datasets/' + dataset + '/active_learning/train/Annotations'
    elif dataset == 'tmp':
        utils.LOG_FILEPATH = r'./checkpoints/active_learning/' + dataset + '/ac_log_' + ac_type + '.txt'

        CLASS = ['__background__', 'car', 'bicycle', 'person', 'bus', 'motorcycle', 'rider', 'truck', 'train']
        train_data_path = r'./datasets/' + dataset + '/active_learning/train'  # 训练集
        test_data_path = r'./datasets/' + dataset + '/active_learning/test'  # 测试集
        output_dir_root = r'./checkpoints/active_learning/' + dataset + '/' + ac_type + '/'

        M = 5
        imgs_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/JPEGImages'
        xmls_file_path = r'./datasets/' + dataset + '/active_learning/unlabelpool/Annotations'
        txt_file_name_root = './each_ite_choose_file_record/' + dataset + '/' + ac_type + '/baseline_' + ac_type + '_5_percent_on_target_'

        to_img_file_path = r'./datasets/' + dataset + '/active_learning/train/JPEGImages'
        to_xml_file_path = r'./datasets/' + dataset + '/active_learning/train/Annotations'

    # 文件路径检查
    if not os.path.exists(output_dir_root):
        os.makedirs(output_dir_root)
    if not os.path.exists(os.path.dirname(txt_file_name_root)):
        os.makedirs(os.path.dirname(txt_file_name_root))
    if not os.path.exists(os.path.dirname(utils.LOG_FILEPATH)):
        os.makedirs(os.path.dirname(utils.LOG_FILEPATH))

    # Active Learning
    for i in range(0, 19):     # 此处应该从当前已有的文件夹序号的下一个序号处开始迭代
        if utils.IF_CREATE_LOG_FILT:
            with open(utils.LOG_FILEPATH, 'a') as af:
                af.write('-------------------------------------------------\n')
                af.write(str(i) + ' - {}%\n'.format(10+i*5))
        # 训练 或 测试
        if i == 0:
            is_first = True  # 是否是初始训练轮次
            model_name = ''  # 当前轮使用的初始模型
        else:
            is_first = False  # 是否是初始训练轮次
            # 获取上一轮的最优模型
            print(i, i-1, max_mAP_idx)
            model_name = file_tools.choosemodel(max_mAP_idx, output_dir_root + str(i-1))
            print(model_name)
        model_path = output_dir_root + str(i-1) + '/' + model_name
        output_dir = output_dir_root + str(i)  # 模型输出路径
        test_model_path = 'xxx'  # 用于测试的模型路径
        valid_path = r'xxx'  # 验证集的路径（路径不能包含中文）
        is_train = True  # 是训练还是测试
        max_mAP, max_mAP_idx = trainer_tmp(model_path, is_train, is_first, test_model_path, valid_path,
                       train_data_path, test_data_path, output_dir, CLASS)  # 记得去 函数内去改参数
        print('--------------------------------')
        print(max_mAP, max_mAP_idx)
        print('--------------------------------')

        # 随机挑选一批数据
        # 每次挑选 unlablepool 中的 5%（of total）数据（也即每次挑选 72 张图片）
        last_file_num = len(file_tools.get_all_file_list(imgs_file_path))
        if last_file_num == 0:
            print('all unlabel data has been labeled!')
            break
        if last_file_num < M * 2:
            M = last_file_num
        imgs_list, xmls_list = chooseStrategy.random_sample(imgs_file_path, xmls_file_path, M)
        for img, xml in zip(imgs_list, xmls_list):
            print(img, xml)

        # 将挑出来的数据存档并转移到 train 或 tmp 中
        txt_file_name = txt_file_name_root + str(time.time()).split('.')[0] + '.txt'
        file_tools.text_save(txt_file_name, imgs_list)
        # 转移文件
        for img, xml in zip(imgs_list, xmls_list):
            shutil.move(img, os.path.join(to_img_file_path, img.split(os.sep)[-1].split('.')[0] + '.jpg'))
            shutil.move(xml, os.path.join(to_xml_file_path, xml.split(os.sep)[-1].split('.')[0] + '.xml'))
        print('------------------转移成功------------------')
