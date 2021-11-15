import os

import torch
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils

image_path = os.path.join(os.getcwd(), 'testdata', '0575.jpg')
model_path = os.path.join(os.getcwd(), 'checkpoints', 'ac_then_gmm', 'model_custom_voc_6.pth')


def main():
    input = []
    src_img = cv2.imread(image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    # torchvision 要求输入的图片数据必须是 list[tensor1, tensor2,...] 格式
    # 且其中图片采用 RGB 格式，tensor 的格式为 [C, H, W]，像素值压缩到 0-1 之间（float）
    # permute(2, 0, 1): [H, W, C] => [C, H, W]
    if torch.cuda.is_available():
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
    else:
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
    input.append(img_tensor)

    # 加载保存的模型(直接加载整个网络，同时附带网络参数，这要求在保存模型时直接保存整个网络)
    # model = torch.load(r'D:\D0\coding\workspaceforC_ide\PycharmCommunity2019\GitHub_Pros\pytorch_fasterrcnn_detection\pytorch_fasterrcnn_detection-master\pytorch_fasterrcnn_detection\save_name_0.pkl')

    # 另一种加载方式（先构建一个与训练得到的模型相同结构的网络模型，然后再将参数加载进网络模型，这要求再保存模型时仅保存训练参数）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(utils.CLASS))
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(f=model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 截取前两层的特征提取层作为 backbone 网络（由于前面已经加载了网络参数，所以这里的参数是已经加载好了的）
    backbone_model = torch.nn.Sequential(*list(model.children())[:2])
    print('-'*30 + ' backbone network struct ' + '-'*30)
    print(backbone_model)
    print('-'*70)

    # 固定网络参数
    backbone_model.eval()
    # get some dummy image
    x = torch.rand(1, 3, 64, 64)
    # compute the output
    output = backbone_model(x)
    print(output)

    # # print(img_tensor.size())
    # # print(input)
    #
    # for i in range(len(input)):
    #     input[i] = input[i].numpy()
    # # input = [t.numpy for t in input]
    # # print(input)
    #
    # input = torch.Tensor(input)
    # print(input.size(), input)
    #
    # print('+'*30 + ' 测试结果 ' + '+' * 30)
    # output = backbone_model(input)
    # print(output)
    # print('+'*70)


def backbonetest():
    input = []
    src_img = cv2.imread(image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    # torchvision 要求输入的图片数据必须是 list[tensor1, tensor2,...] 格式
    # 且其中图片采用 RGB 格式，tensor 的格式为 [C, H, W]，像素值压缩到 0-1 之间（float）
    # permute(2, 0, 1): [H, W, C] => [C, H, W]
    if torch.cuda.is_available():
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
    else:
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
    input.append(img_tensor)

    # 另一种加载方式（先构建一个与训练得到的模型相同结构的网络模型，然后再将参数加载进网络模型，这要求再保存模型时仅保存训练参数）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(utils.CLASS))
    if torch.cuda.is_available():
        model.cuda()
    checkpoint = torch.load(f=model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    # 截取 faster-rcnn 中的 backbone 网络（由于前面已经加载了网络参数，所以这里的参数是已经加载好了的）
    backbone = model.backbone
    # print('-' * 30 + ' backbone network struct ' + '-' * 30)
    # print(backbone)
    # print('-' * 70)
    print('-' * 30 + ' backbone net name and parameters ' + '-' * 30)
    for name, parameter in backbone.named_parameters():
        print(name, ':', parameter)
    print('-' * 70)

    # 固定网络参数（这一句不加，backbone 网络的参数也不会改变！）
    # backbone.eval()
    # get some dummy image
    x = torch.rand(1, 3, 64, 64)
    # compute the output
    output = backbone(x)
    pool256 = output['pool']
    print(pool256.size())
    pool256_onedim = pool256.view(pool256.size(0), -1)  # 压缩成一维
    print(pool256_onedim.size())
    pool256_list = pool256_onedim.detach().numpy()
    print(pool256_list)

    # 看一下网络参数有没有改变
    print('-' * 30 + ' backbone net name and parameters ' + '-' * 30)
    for name, parameter in backbone.named_parameters():
        print(name, ':', parameter)
    print('-' * 70)


def backbone_only():
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    backbone = resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=3)
    print('-'*30 + ' backbone net struct ' + '-'*30)
    print(backbone)
    print('-'*70)
    print('-' * 30 + ' backbone net layer info ' + '-' * 30)
    for name, parameter in backbone.named_parameters():
        print(name, ":", parameter.size())
    print('-' * 70)

    # get some dummy image
    x = torch.rand(1, 3, 64, 64)
    # compute the output
    output = backbone(x)
    # print(output)
    print([(k, v.shape) for k, v in output.items()])
    # 这里需要注意的是，resnet50 共有5个 layer（具体可参见：https://blog.csdn.net/lanran2/article/details/79057994#commentBox），
    # 这 5 个layer 都是进行卷积操作的，最后有一个 average pool 层，用于将前面的输入变成 2048 维的输出，用于后面的分类和回归。
    # 但是本代码中的 backbone 后面自动的加了 fpn 层，将多层特征进行了融合，最后输出 256 维的特征。
    # 取出最后的平均池化后的结果
    pool256 = output['pool']
    print(pool256.size(), pool256)      # torch.Size([1, 256, 1, 1])
    pool256_onedim = pool256.view(pool256.size(0), -1)  # 将 256 维压缩到一维
    print(pool256_onedim)
    # tensor 转换成普通数组
    import numpy as np
    pool256_list = pool256_onedim.detach().numpy()
    print(pool256_list)
    """
    returns
    # [('0', torch.Size([1, 256, 16, 16])),
    #      ('1', torch.Size([1, 256, 8, 8])),
    # ('2', torch.Size([1, 256, 4, 4])),
    # ('3', torch.Size([1, 256, 2, 2])),
    # ('pool', torch.Size([1, 256, 1, 1]))]
    """


if __name__ == '__main__':
    # main()
    backbonetest()
    # backbone_only()
