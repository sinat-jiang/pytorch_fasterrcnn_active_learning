from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import xml.etree.ElementTree as ET
import os
import shutil
import pickle
import numpy as np
import pdb
from torchvision.datasets import VOCDetection

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image
import random
import cv2
# import matplotlib; matplotlib.use('Qt5Agg')
import utils


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        # obj_struct['pose'] = obj.find('pose').text
        # obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(classname,
             detpath,
             imagesetfile,
             annopath='',
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections （使用当前模型进行预测，存放对应图像的预测结果文件）
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations   （对应图像的真实标注文件，即 gt）
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    recs = {}
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        # load annotations
        for i, imagename in enumerate(imagenames):
            # 解析 xml 标注文件，获取对应图片的名称以及图片中的所有 bbox 参数
            recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        # R 用来只保存 指定类(classname) 的 bbox
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def custom_voc_eval(classname,
                    detpath,
                    imagesetfile,
                    annopath='',
                    ovthresh=0.5,
                    use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations xml标准文件路径,一般在Annotations里面
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.只包含图片名称的文本文件
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    recs = {}
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
        imagenames = [x.strip() for x in lines]

        # load annotations
        for i, imagename in enumerate(imagenames):
            # print(annopath.format(imagename))  # 这里的格式为：xxx/xx/x0160.xml
            # print(imagename)
            # print(imagesetfile)  # E:\AllDateSets\MilitaryOB_5_class_torch\test\imagesetfile.txt
            # 这里需要将文件路径补全
            recs[imagename] = parse_rec(os.path.join(
                os.path.dirname(imagesetfile), 'Annotations', os.path.basename(annopath.format(imagename))))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    if BB.shape[0] > 0:
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def _write_voc_results_file(all_boxes, image_index, root, classes):
    """将模型对测试集图像的预测结果存到文件中（包含图像名称、预测的 bbox）"""
    if os.path.exists('/tmp/results'):
        shutil.rmtree('/tmp/results')
    os.makedirs('/tmp/results')
    print('Writing results file', end='\r')

    for cls_ind, cls in enumerate(classes):
        # DistributeSampler happens to clone the inputs to make the task 
        # lenghts even among the nodes:
        # https://github.com/pytorch/pytorch/issues/22584
        # Boxes can be duplicated in the process since multiple
        # evaluation of the same image can happen, multiple boxes in the
        # same location decrease the final mAP, later in the code we discard
        # repeated image_index thanks to the sorting
        new_image_index, all_boxes[cls_ind] = zip(*sorted(zip(image_index,
                                                              all_boxes[cls_ind]), key=lambda x: x[0]))
        if cls == '__background__':
            continue
        # images_dir = data_loader.dataset.image_dir
        filename = '/tmp/results/det_test_{:s}.txt'.format(cls)
        with open(filename, 'wt') as f:
            prev_index = ''
            for im_ind, index in enumerate(new_image_index):
                # check for repeated input and discard
                if prev_index == index: continue
                prev_index = index
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                dets = dets[0]
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def _write_custom_voc_results_file(data_loader, all_boxes, image_index, root, classes, thread=0.8):
    if os.path.exists('./tmp/results'):
        # 递归地删除文件
        shutil.rmtree('./tmp/results')
    os.makedirs('./tmp/results')
    print('Writing results file', end='\r')

    os.makedirs("output", exist_ok=True)  # 创建ｏｕｔｐｕｔ目录，存储图片检测结果
    # Bounding-box colors
    # cmap = plt.get_cmap("tab20b")
    # colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    # 这里的 colors 要按照自己的
    # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 255), (0, 0, 255)]
    colors = [random_color() for i in range(len(classes))]

    for cls_ind, cls in enumerate(classes):
        # DistributeSampler happens to clone the inputs to make the task 
        # lenghts even among the nodes:
        # https://github.com/pytorch/pytorch/issues/22584
        # Boxes can be duplicated in the process since multiple
        # evaluation of the same image can happen, multiple boxes in the
        # same location decrease the final mAP, later in the code we discard
        # repeated image_index thanks to the sorting
        new_image_index, all_boxes[cls_ind] = zip(*sorted(zip(image_index,
                                                              all_boxes[cls_ind]), key=lambda x: x[0]))
        if cls == '__background__':
            continue
        images_dir = data_loader.dataset.image_dir
        filename = './tmp/results/det_test_{:s}.txt'.format(cls)

        with open(filename, 'wt') as f:
            prev_index = ''
            for im_ind, index in enumerate(new_image_index):
                # opencv读取图片（注意这里的路径要求不能有中文）
                img = cv2.imread(os.path.join(images_dir, index + '.jpg'))
                # 打印出测试的图片
                # print(os.path.join(images_dir, index + '.jpg'))
                h, w, _ = img.shape

                # check for repeated input and discard
                if prev_index == index: continue
                prev_index = index
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                dets = dets[0]

                # bbox_colors = random1.sample(colors, 3)

                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))
                    if dets[k, -1] < thread:
                        continue
                    # print("\t+ Label: %s, Conf: %.5f" % (cls, dets[k, -1]))
                    x1, x2 = dets[k, 0], dets[k, 2]
                    y1, y2 = dets[k, 1], dets[k, 3]
                    box_w = x2 - x1
                    box_h = y2 - y1

                    color = colors[cls_ind]
                    thick = int((h + w) / 300)
                    # print(os.path.join(images_dir, index + '.jpg'))     # 00002968
                    # print(img, (x1, y1), (x2, y2), color, img, (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item())), color, thick)
                    cv2.rectangle(img, (int(x1.item()), int(y1.item())), (int(x2.item()), int(y2.item())), color, thick)
                    mess = '%s: %.3f' % (cls, dets[k, -1])
                    cv2.putText(img, mess, (int(x1.item()), int(y1.item()) - 12), 0, 1e-3 * h, color, thick // 3)

                filename = index
                cv2.imwrite(f"output/output-{filename}.png", img)


def _do_python_eval(data_loader):
    imagesetfile = os.path.join(data_loader.dataset.root,
                                'VOCdevkit/VOC2007/ImageSets/Main/' + data_loader.dataset.image_set + '.txt')
    annopath = os.path.join(data_loader.dataset.root,
                            'VOCdevkit/VOC2007/Annotations/{:s}.xml')

    classes = data_loader.dataset._transforms.transforms[0].CLASSES
    aps = []
    for cls in classes:
        if cls == '__background__':
            continue
        filename = '/tmp/results/det_test_{:s}.txt'.format(cls)
        rec, prec, ap = voc_eval(cls, filename, imagesetfile, annopath,
                                 ovthresh=0.5, use_07_metric=True)
        print('+ Class {} - AP: {}, precision: {}, recall: {}'.format(cls, ap, prec, rec))
        aps += [ap]
    print('Mean AP = {:.4f}        '.format(np.mean(aps)))


def _do_python_eval_custom_voc(data_loader, classes, use_07_metric=True):
    imagesetfile = os.path.join(data_loader.dataset.root, 'imagesetfile.txt')
    annopath = os.path.join(data_loader.dataset.annotation_dir, '{:s}.xml')

    # classes = data_loader.dataset._transforms.transforms[0].CLASSES
    aps = []
    # 若要展示 P-R 曲线，将注释去掉（每个 epoch 结束后都会展示）
    # fig = plt.figure()

    retStr = ''
    for cls in classes:
        if cls == '__background__':
            continue
        filename = './tmp/results/det_test_{:s}.txt'.format(cls)
        rec, prec, ap = custom_voc_eval(cls, filename, imagesetfile, annopath,
                                        ovthresh=0.5, use_07_metric=use_07_metric)
        print('+ Class {} - AP: {}'.format(cls, ap))
        retStr += '+ Class {} - AP: {}'.format(cls, ap) + '\n'
        plt.plot(rec, prec, label='{}'.format(cls))
        aps += [ap]
    # plt.xlabel('recall')
    # plt.ylabel('precision')
    # plt.legend()
    # plt.show()
    print('Mean AP = {:.4f}        '.format(np.mean(aps)))
    retStr += 'Mean AP = {:.4f}        '.format(np.mean(aps)) + '\n'
    # 将 mAP 返回,用于挑选当前迭代轮次的最优模型
    return np.mean(aps), retStr


def random_color():
    """随机取色（三原色）"""
    import random
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)
