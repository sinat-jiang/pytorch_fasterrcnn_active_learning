import os
import numpy as np
import torch
from PIL import Image
os.chdir('/home/gaoya/Files/python/pytorch/torchvision_voc/references/detection')

import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator

from engine import train_one_epoch, voc_evaluate
from voc_utils import get_voc,get_custom_voc,VOCCustomData, ConvertCustomVOCtoCOCO
import utils
import transforms as T

from torchvision.datasets import VOCDetection
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class VOCDetection_flip(VOCDetection):
	def __init__(self, img_folder, year, image_set, transforms):
		super().__init__(img_folder, year, image_set)
		self._transforms = transforms

	def __getitem__(self, idx):
		real_idx = idx//2
		img, target = super(VOCDetection_flip, self).__getitem__(real_idx)
		target = dict(image_id=real_idx, annotations=target['annotation'])
		if self._transforms is not None:
			img, target = self._transforms(img, target)
			# img = img[[2, 1, 0],:]

			if (idx % 2) == 0:
				height, width = img.shape[-2:]
				img = img.flip(-1)
				bbox = target["boxes"]
				bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
				target["boxes"] = bbox

		return img, target

	def __len__(self):
		return 2*len(self.images)

# def get_voc(root, transforms, image_set='train',  year = '2007', download=False):
# 	t = [PrepareInstance()]

# 	if transforms is not None:
# 		t.append(transforms)
# 	transforms = T.Compose(t)

# 	dataset = VOCDetection(root,  image_set=image_set, transforms=transforms, year='2007', download=download)

# 	return dataset

def get_transform(istrain=False):
	transforms = []
	transforms.append(T.ToTensor())
	if istrain:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)

class BoxHead(nn.Module):
	def __init__(self, vgg):
		super(BoxHead, self).__init__()
		self.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
		self.in_features = 4096 # feature out from mlp

	def forward(self, x):
		x = x.flatten(start_dim=1)
		x = self.classifier(x)
		return x


def get_model_FRCNN(num_classes):

	backbone = torchvision.models.mobilenet_v2(pretrained=True).features
	backbone.out_channels = 1280
	anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
								aspect_ratios=((0.5, 1.0, 2.0),))
	roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
													output_size=7,
													sampling_ratio=2)
	model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone,
					num_classes,
					rpn_anchor_generator=anchor_generator,
					box_roi_pool=roi_pooler)

	return model


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 3 	# 20 classes + background for VOC
BATCH_SIZE = 2

train_dataset_root = '/home/gaoya/Documents/quexianjiance/GAN.NO2/train_GAN_twoclass'
dataset = get_custom_voc(root=train_dataset_root, transforms=get_transform(istrain=True))

test_dataset_root = '/home/gaoya/Documents/quexianjiance/GAN.NO2/test_clear_algorthmblur'
dataset_test = get_custom_voc(root=test_dataset_root, transforms=get_transform(istrain=False))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
	dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
	collate_fn=utils.collate_fn)


data_loader_test = torch.utils.data.DataLoader(
	dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
	collate_fn=utils.collate_fn)

print('data prepared, train data: {}'.format(len(dataset)))
print('data prepared, test data: {}'.format(len(dataset_test)))

#%%
# model = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes,
                                                            #   pretrained='pretrained')
# model.to(device)

# get the model using our helper function
model = get_model_FRCNN(num_classes)
#
# move model to the right device
model.to(device)
#
# # construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
							momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
										step_size=30,
										gamma=0.1)

# let's train it for 10 epochs
num_epochs = 40

# setup log data writer
if not os.path.exists('log'):
	os.makedirs('log')
# writer = SummaryWriter(log_dir='log')
#
#%% Start training!
iters_per_epoch = int(len(data_loader) / data_loader.batch_size)
for epoch in range(num_epochs):
	loss_epoch = {}
	loss_name = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
	for ii, (images, targets) in tqdm(enumerate(data_loader), total=len(data_loader)):
		model.train()
		optimizer.zero_grad()
		images = list(image.to(device) for image in images)
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
		# training
		loss_dict = model(images, targets)
		losses = sum(loss for loss in loss_dict.values())
		losses.backward()
		optimizer.step()
		lr_scheduler.step()
		info = {}
		for name in loss_dict:
			info[name] = loss_dict[name].item()

		# writer.add_scalars("losses", info, epoch * iters_per_epoch + ii)

	if (epoch + 1) % 1 == 0:
		# evaluate on the test dataset
		voc_evaluate(model, data_loader_test, device=device)

torch.save(model, './torchvision_faster_rcnn_pascal.pth')
# writer.close()