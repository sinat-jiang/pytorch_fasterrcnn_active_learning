"""使用训练好的网络来进行预测"""
import os

import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import utils


def prediction_img(model_path, image_path='./testdata/0000.jpg'):
    """输入单张图片，利用训练好的模型进行预测"""
    input = []
    src_img = cv2.imread(image_path)
    img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    # torchvision 要求输入的图片数据必须是 list[tensor1, tensor2,...] 格式
    # 且其中图片采用 RGB 格式，tensor 的格式为 [C, H, W]，像素值压缩到 0-1 之间（float）
    # permute(2, 0, 1): [H, W, C] => [C, H, W]
    img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
    input.append(img_tensor)

    # 加载保存的模型(直接加载整个网络，同时附带网络参数，这要求在保存模型时直接保存整个网络)
    # model = torch.load(r'D:\D0\coding\workspaceforC_ide\PycharmCommunity2019\GitHub_Pros\pytorch_fasterrcnn_detection\pytorch_fasterrcnn_detection-master\pytorch_fasterrcnn_detection\save_name_0.pkl')

    # 另一种加载方式（先构建一个与训练得到的模型相同结构的网络模型，然后再将参数加载进网络模型，这要求再保存模型时仅保存训练参数）
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(utils.CLASS))
    model.cuda()
    checkpoint = torch.load(f=model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    model.eval()

    out = model(input)
    print(out)

    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']
    print(boxes)
    print(labels)
    print(scores)

    # 数据集的类别（包括背景类）
    names = utils.CLASS

    # 返回一堆的检测框，需要手动进行阈值选取，确定最后的 bbox
    for idx in range(boxes.shape[0]):
        # 返回一堆的 bbox，但是仅筛选出超过指定阈值（args.score）的 bbox
        if scores[idx] >= 0.8:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = names[labels[idx].item()]
            score = str(round(scores[idx].item(), 3))  # round(2.3456, 3) = 2.345  即保留小数点后 3 位
            # 画上检测框
            cv2.rectangle(src_img, (x1, y1), (x2, y2), random_color(), thickness=2)
            # 添上类别说明
            cv2.putText(src_img, text=name + ' ' + score, org=(x1, y1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 255, 255))

    cv2.imshow('result', src_img)
    cv2.imwrite(filename=image_path[:-4] + '-detected.jpg', img=src_img)  # 保存图片
    cv2.waitKey()
    cv2.destroyAllWindows()


def random_color():
    """随机取色（三原色）"""
    import random
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


if __name__ == '__main__':
    # 使用现有模型对单张图片进行预测
    path = os.path.join(os.getcwd(), 'checkpoints', 'model_custom_voc_0_loss_0.28757190704345703.pth')
    prediction_img(image_path='./testdata/0000.jpg', model_path=path)
