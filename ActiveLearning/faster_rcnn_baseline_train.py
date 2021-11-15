"""
仅使用 faster_rcnn 进行训练，作为基准实验结果
"""
import datetime
import os
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import sys
sys.path.append("..")
import utils
from engine import custom_voc_evaluate, train_one_epoch
from train_data_pro_def import get_dataset, get_transform


def train_or_test(args):

    if not args['is_first_train']:  # 不是第一次训练时，不要加载预训练模型
        args['pretrained'] = False

    device = torch.device(args['device'])
    # Data loading code
    print("Loading data")
    # 如果是自定义Pascal数据集,不需要传入image_set参数,因此这里设置为None
    dataset, num_classes = get_dataset(get_transform(train=True), args['train_data_path'], CLASS=args['CLASS'])
    dataset_test, _ = get_dataset(get_transform(train=False), args['test_data_path'], CLASS=args['CLASS'])

    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=args['workers'],
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=args['workers'],
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=args['pretrained'])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 多 GPU 训练通过下面语句实现
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model)
    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])

    # 等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step_size。间隔单位是step。需要注意的是， step 通常是指 epoch，不要弄成 iteration 了
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # 按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args['lr_steps'], gamma=args['lr_gamma'])

    start_epoch = 0
    if not args['is_first_train']:      # 如果不是第一次训练，需要恢复训练
        print('--------------load last model parameters--------------')
        checkpoint = torch.load(args['resume'], map_location='cpu')
        start_epoch = int(os.path.basename(args['resume']).split('_')[3]) + 1
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])  # 用于恢复训练,处理模型还需要优化器和学习率规则
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

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
    for epoch in range(start_epoch, args['epochs']):
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

        mAP, retStr = custom_voc_evaluate(model, data_loader_test, device=device, CLASS=args['CLASS'], thresh=1)
        if utils.LOG_FILEPATH_FOR_FASTER_RCNN is None:
            pass
        else:
            with open(utils.LOG_FILEPATH_FOR_FASTER_RCNN, 'a') as f:
                f.write('----------------------------------' + 'epoch: ' + str(epoch) + '----------------------------------\n')
                f.write(retStr)
                f.write('\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def param_define_and_going(train_data_path, test_data_path, output_path, is_train, is_first, test_model_path, model_path, CLASS):
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
    args['lr_steps'] = [24, 32]
    args['lr_gamma'] = 0.1
    args['epochs'] = 1  # 训练的轮次
    args['print_freq'] = 20
    args['output_dir'] = output_path  # 模型输出路径
    # 训练
    args['resume'] = None
    args['test_only'] = False
    args['pretrained'] = False  # 是否使用 coco 上预训练的模型
    # args['dist-url'] = 'env://'
    # args['world-size'] = 3

    if not is_train:
        # 测试
        args['pretrained'] = False  # 都是测试了，肯定不会使用 coco 上预训练的模型，而是用自己的模型
        # resume 是保存的网络的各项参数的文件名
        args['resume'] = os.path.join(os.getcwd(), 'checkpoints', test_model_path)
        args['test_only'] = True
        args['test_data_path'] = test_data_path
        args['is_first_train'] = False
    else:
        # 若是后面的轮次训练，记得不使用预训练模型，但是需要使用上次训练好的参数
        args['is_first_train'] = is_first
        if is_first:
            args['resume'] = None
        else:
            args['resume'] = os.path.join(os.getcwd(), 'checkpoints', model_path)
        # 每次都连续训练 10 次，选取其中 mAP 最高的模型保存
        args['epochs'] = 40  # 不是第一次，之后的迭代训练也要保证所有类的 AP 在指定阈值之上
    train_or_test(args)


if __name__ == '__main__':
    # 训练
    is_train = True
    is_first = True
    test_model_path = r'xxx'
    # model_path = r'E:\ac_pro_code_editions\2021-11-14\pytorch_fasterrcnn_active_learning\ActiveLearning\checkpoints\faster_rcnn_baseline\tmp\model_custom_voc_3_loss_0.8405036330223083.pth'     # 测试 or 断点续训时指定的上一个 model
    model_path = r'xxx'

    # dataset = 'sim10k'
    # dataset = 'cityscape'
    dataset = 'cityscapefoggy'
    dataset = 'tmp'
    train_data_path = None
    test_data_path = None
    output_path = None
    CLASS = None

    if dataset == 'cityscape':
        train_data_path = r'./datasets/' + dataset + '/faster_rcnn_baseline/train'
        test_data_path = r'./datasets/' + dataset + '/faster_rcnn_baseline/test'
        output_path = r'./checkpoints/faster_rcnn_baseline/' + dataset
        CLASS = ["__background__", 'person', 'car', 'motorcycle']
    if dataset == 'cityscape_single_obj':
        train_data_path = r'./datasets/' + dataset + '/faster_rcnn_baseline/train'
        test_data_path = r'./datasets/' + dataset + '/faster_rcnn_baseline/test'
        output_path = r'./checkpoints/faster_rcnn_baseline/' + dataset
        CLASS = ["__background__", 'car']
    elif dataset == 'cityscapefoggy':
        train_data_path = r"./datasets/" + dataset + "/faster_rcnn_baseline/train"
        test_data_path = r"./datasets/" + dataset + "/faster_rcnn_baseline/test"
        output_path = r'./checkpoints/faster_rcnn_baseline/' + dataset
        CLASS = ['__background__', 'car', 'bicycle', 'person', 'bus', 'motorcycle', 'rider', 'truck', 'train']
    elif dataset == 'sim10k':
        train_data_path = r"./datasets/" + dataset + "/faster_rcnn_baseline/train"
        test_data_path = r"./datasets/" + dataset + "/faster_rcnn_baseline/test"
        output_path = r'./checkpoints/faster_rcnn_baseline/' + dataset
        CLASS = ["__background__", 'person', 'car', 'motorcycle']
    elif dataset == 'tmp':
        train_data_path = r"./datasets/" + dataset + "/faster_rcnn_baseline/train"
        test_data_path = r"./datasets/" + dataset + "/faster_rcnn_baseline/test"
        output_path = r'./checkpoints/faster_rcnn_baseline/' + dataset
        CLASS = ['__background__', 'car', 'bicycle', 'person', 'bus', 'motorcycle', 'rider', 'truck', 'train']

    # 文件夹检查
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # log 记录文件
    utils.LOG_FILEPATH_FOR_FASTER_RCNN = os.path.join(output_path, 'faster_rcnn_log.txt')

    # 测试
    # train_data_path = r"/home/x/D2_acl_tl/ATL/cityspace_atl/train-on-source-test-on-target/source/train"
    # test_data_path = r'/home/x/D2_acl_tl/ATL/cityspace_atl/train-on-source-test-on-target/target_foggy'
    # output_path = r'xxx'
    # is_train = False
    # is_first = True
    # test_model_path = r'acl_tl/cityspace_real2foggy/train_on_source/model_custom_voc_15_loss_0.18987548351287842.pth'
    # model_path = r'xxx'

    param_define_and_going(train_data_path, test_data_path, output_path, is_train, is_first, test_model_path, model_path, CLASS)    # 记得去函数内去改参数
