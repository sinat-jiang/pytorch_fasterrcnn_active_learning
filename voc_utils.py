import torch
import torchvision
import os
import sys
import tarfile

import transforms as T
import collections

import utils

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from PIL import Image
import pandas as pd


class ConvertVOCtoCOCO(object):
    CLASSES = (
        "__background__", "aeroplane", "bicycle",
        "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog",
        "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor",
    )

    def __call__(self, image, target):
        # return image, target
        anno = target['annotations']
        filename = anno["filename"].split('.')[0]
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        ishard = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            ishard.append(int(obj['difficult']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        ishard = torch.as_tensor(ishard)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ishard"] = ishard
        target['name'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8) #convert filename in int8

        return image, target


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, img_folder, year, image_set, transforms, download):
        super(VOCDetection, self).__init__(img_folder,  year, image_set,download)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super(VOCDetection, self).__getitem__(idx)
        target = dict(image_id=idx, annotations=target['annotation'])
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            # img = img[[2, 1, 0],:]
        return img, target


def get_voc(root, image_set, transforms):
    t = [ConvertVOCtoCOCO()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = VOCDetection(img_folder=root, year='2007', image_set=image_set, transforms=transforms,download=True)

    return dataset


class VOCCustomData(torchvision.datasets.vision.VisionDataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.

    Args:
        root (string): Root directory of the custom VOC Dataset which includes directories
            Annotations and JPEGImages

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 transforms=None,
                 CLASS=None):
        super(VOCCustomData, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        self.CLASS = CLASS
        self._transforms = transforms

        voc_root = self.root
        self.image_dir = os.path.join(voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(voc_root, 'Annotations')

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' Please verify the correct Dataset!')
        file_names = []

        for imgs in os.listdir(self.image_dir):
            file_names.append(imgs.split('.')[0])

        images_file = pd.DataFrame(file_names, index=None)
        # 保存图像路径,注意只有文件名,不带后缀和文件路径
        images_file.to_csv(voc_root+'/imagesetfile.txt', header=False, index=False)

        self.images = [os.path.join(self.image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(self.annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')

        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        target = dict(image_id=index, annotations=target['annotation'])

        if self.transforms is not None:
            img, target = self.transforms(img, target, self.CLASS)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


class ConvertCustomVOCtoCOCO(object):
    # def __init__(self):
    #     pass

    def __call__(self, image, target, CLASS):
        # return image, target
        anno = target['annotations']
        filename = anno["filename"].split('.')[0]
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        ishard = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            #
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(CLASS.index(obj['name']))
            ishard.append(int(obj['difficult']))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        ishard = torch.as_tensor(ishard)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["ishard"] = ishard
        target['name'] = torch.tensor([ord(i) for i in list(filename)], dtype=torch.int8) #convert filename in int8

        return image, target


def get_custom_voc(root, transforms, CLASS):
    t = [ConvertCustomVOCtoCOCO()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    dataset = VOCCustomData(root=root, transforms=transforms, CLASS=CLASS)

    return dataset