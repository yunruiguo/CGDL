##################################################
# Imports
##################################################

import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from PIL import Image
from flexilabelconvert import get_mnist_svhn_cifar10_splits, get_cifar10plus_splits, get_cifar50plus_splits, get_tiny_imagenet_splits
import subprocess
import os


##################################################
# Params
##################################################

DATASET_CONFIGS = {
	'mnist': {'size': 28, 'channels': 1, 'classes': 6},
	'svhn': {'size': 32, 'channels': 3, 'classes': 6},
	'cifar10': {'size': 32, 'channels': 3, 'classes': 6},
	'cifar+10': {'size': 32, 'channels': 3, 'classes': 4},
	'cifar+50': {'size': 32, 'channels': 3, 'classes': 4},
	'tiny_imagenet': {'size': 32, 'channels': 3, 'classes': 20},
}

##################################################
# DatasetSubset class
##################################################

class DatasetSubset(torch.utils.data.Dataset):
	def __init__(self, ds, classes, return_real_classes=False, return_outlier_classes=False):
		super(DatasetSubset, self)
		self.ds = ds
		self._classes = classes
		self.idxs = self._get_subset_sample_idxs()
		self.return_real_classes = return_real_classes
		self.return_outlier_classes = return_outlier_classes
	def _get_subset_sample_idxs(self):
		idxs = [i for i, (x, y) in enumerate(self.ds) if y in self._classes]
		return idxs

	def __getitem__(self, idx):
		idx_ds = self.idxs[idx]
		x, y = self.ds.__getitem__(idx_ds)
		if self.return_real_classes:
			return x, y
		elif self.return_outlier_classes:
			return x, -1
		else:
			return x, self._classes.index(y)

	def __len__(self):
		return len(self.idxs)


##################################################
# MergeDatasets class
##################################################

class MergeDatasets(torch.utils.data.Dataset):
	def __init__(self, ds_list):
		super(MergeDatasets, self).__init__()
		self.ds_list = ds_list
		self.ds_lengths = [len(ds) for ds in self.ds_list]
		self._classes = [] 
		for ds in self.ds_list:
			for cl in ds._classes:
				if cl not in self._classes:
					self._classes += [cl]

	def __len__(self):
		length = sum(self.ds_lengths)
		return length

	def __getitem__(self, idx):
		if idx >= len(self): raise IndexError
		length_cum = [0]
		for l in self.ds_lengths:
			length_cum += [length_cum[-1] + l]

		for i_l, l in enumerate(length_cum[1:], start=1):
			if (length_cum[i_l - 1] <= idx) and (idx < l):
				ds = self.ds_list[i_l - 1]
				idx_ds = idx - length_cum[i_l - 1]
				break
		return ds.__getitem__(idx_ds)



##################################################
# TinyImagenetDataset class
##################################################

class TinyImagenetDataset(torch.utils.data.Dataset):
	def __init__(self, data_dir, train=True, transforms=None, download=False):
		super(TinyImagenetDataset, self).__init__()
		self.data_dir = data_dir
		self.train = train
		self.transforms = transforms
		if download:
			self._download()
		self.labels_list = self._retrieve_labels_list()
		self.image_paths, self.labels = self._get_data()

	def _download(self):
		url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
		subprocess.run(f'wget -r -nc -P {self.data_dir} {url}'.split())
		subprocess.run(f'unzip -qq -n {self.data_dir}/cs231n.stanford.edu/tiny-imagenet-200.zip -d {self.data_dir}'.split())

	def _retrieve_labels_list(self):
		labels_list = []
		with open(f'{self.data_dir}/tiny-imagenet-200/wnids.txt', 'r') as f:
			for line in f.readlines():
				line = line.strip()
				if len(line) > 0:
					labels_list += [line]
		return labels_list

	def _get_data(self):
		image_paths, labels = [], []

		# If train
		if self.train:
			for cl_folder in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train')):
				label = self.labels_list.index(cl_folder)
				for image_name in sorted(os.listdir(f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images')):
					image_path = f'{self.data_dir}/tiny-imagenet-200/train/{cl_folder}/images/{image_name}'
					image_paths += [image_path]
					labels += [label]

		# If validation
		else:
			with open(f'{self.data_dir}/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
				for line in f.readlines():
					line = line.strip()
					if len(line) == 0:
						continue
					image_name, label_str = line.split('\t')[:2]
					image_path = f'{self.data_dir}/tiny-imagenet-200/val/images/{image_name}'
					label = self.labels_list.index(label_str)
					image_paths += [image_path]
					labels += [label]
		return image_paths, labels

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		img = Image.open(self.image_paths[idx])
		if self.transforms is not None:
		    img = self.transforms(img)
		label = self.labels[idx]
		return img, label

class GraImg2ColorImg(object):
    def __init__(self):
        pass

    def __call__(self, x):
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return x


##################################################
# Train and test dataloaders functions
##################################################

def get_train_valid_loader(data_dir,
						   dataset_name,
						   batch_size,
						   num_workers=4,
						   pin_memory=False,
						   split='split0'
						   ):

	data_dir = data_dir + '/' + dataset_name

	# Select train and validation dataset
	# MNIST
	if dataset_name == "mnist":
		trans = [transforms.Resize((28, 28)),
				 transforms.ToTensor()]
		dataset = datasets.MNIST(data_dir, train=True, transform=transforms.Compose(trans), download=True)
		valid_set = datasets.MNIST(data_dir, train=False, transform=transforms.Compose(trans), download=True)
		class_list, _ = get_mnist_svhn_cifar10_splits(dataset.classes, split)
	# SVHN
	elif dataset_name == "svhn":
		trans = [transforms.RandomCrop(28, padding=4),
				 transforms.ToTensor()]
		dataset = datasets.SVHN(data_dir, split='train', download=True,	transform=transforms.Compose(trans))
		trans = [transforms.Resize((28, 28)),
				 transforms.ToTensor()]
		valid_set = datasets.SVHN(data_dir, split='test', download=True,transform=transforms.Compose(trans))
		class_list, _ = get_mnist_svhn_cifar10_splits(dataset.classes, split)
	# CIFAR-10
	elif (dataset_name == "cifar10") or (dataset_name == "cifar+10") or (dataset_name == "cifar+50"):
		trans = [
			transforms.RandomCrop(28, padding=4),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ToTensor(),
		]
		dataset = datasets.CIFAR10(data_dir, train=True, download=True,
				transform=transforms.Compose(trans))
		trans = [transforms.Resize((28, 28)),
			transforms.ToTensor(),
		]
		valid_set = datasets.CIFAR10(data_dir, train=False, download=True,
				transform=transforms.Compose(trans))

		test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=transforms.Compose(trans))
		if dataset_name == 'cifar10':
			class_list, unknown_classes = get_mnist_svhn_cifar10_splits(dataset.classes, split)
		elif dataset_name == 'cifar+10':
			class_list = get_cifar10plus_splits(cifar10_classes=dataset.classes, split=split)
		elif dataset_name == 'cifar+50':
			class_list = get_cifar50plus_splits(cifar10_classes=dataset.classes, split=split)

		else:
			print('error')
	# Tiny-ImageNet
	elif dataset_name == "tiny_imagenet":
		trans = [
			transforms.Resize((28, 28)),
			transforms.RandomCrop(28, padding=4),
			transforms.RandomHorizontalFlip(0.5),
			transforms.ToTensor(),
            GraImg2ColorImg()
		]
		dataset = TinyImagenetDataset(data_dir, train=True, download=True,
			transforms=transforms.Compose(trans))
		trans = [
			transforms.Resize((28, 28)),
			transforms.ToTensor(),
            GraImg2ColorImg(),
		]
		valid_set = TinyImagenetDataset(data_dir, train=False, download=True,
			transforms=transforms.Compose(trans))
		class_list, unknown_classes = get_tiny_imagenet_splits(dataset.labels_list, split)

	dataset = DatasetSubset(dataset, class_list)
	valid_set = DatasetSubset(valid_set, class_list)	

	train_loader = torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, shuffle=True,
		num_workers=num_workers, pin_memory=pin_memory,
	)

	valid_loader = torch.utils.data.DataLoader(
		valid_set, batch_size=batch_size, shuffle=False,
		num_workers=num_workers, pin_memory=pin_memory,
	)

	return train_loader, valid_loader

def get_test_loader(data_dir,
					dataset_name,
					batch_size,
					num_workers=4,
					pin_memory=False,
					split='split0'
					):

	data_dir = data_dir + '/' + dataset_name

	# Select dataset
	# MNIST
	if dataset_name == 'mnist':
		trans = [
			transforms.Resize((28, 28)),
			transforms.ToTensor(),
		]
		dataset_train = datasets.MNIST(data_dir, train=True, transform=transforms.Compose(trans), download=True)
		dataset = datasets.MNIST(data_dir, train=False, transform=transforms.Compose(trans), download=True)
		dataset.classes = list(range(200))
		class_list, unknown_classes = get_mnist_svhn_cifar10_splits(dataset.classes, split)
		dataset = DatasetSubset(dataset, class_list + unknown_classes)
	# SVHN
	elif dataset_name == "svhn":
		trans = [transforms.Resize((28, 28)),
				 transforms.ToTensor()]
		dataset_train = datasets.SVHN(data_dir, split='train', download=True,
								transform=transforms.Compose(trans))
		dataset = datasets.SVHN(data_dir, split='test', download=True,
								transform=transforms.Compose(trans))
		class_list, unknown_classes = get_mnist_svhn_cifar10_splits(dataset.classes, split)
		dataset = DatasetSubset(dataset, class_list + unknown_classes)
	# CIFAR-10
	elif dataset_name == "cifar10":
		trans = [transforms.Resize((28, 28)),
			transforms.ToTensor(),
		]
		dataset_train = datasets.CIFAR10(data_dir, train=True, download=False,
				transform=transforms.Compose(trans))
		dataset = datasets.CIFAR10(data_dir, train=False, download=False,
				transform=transforms.Compose(trans))
		class_list, unknown_classes = get_mnist_svhn_cifar10_splits(dataset.classes, split)
		dataset = DatasetSubset(dataset, class_list + unknown_classes)
	# CIFAR+10
	elif (dataset_name == "cifar+10") or (dataset_name == "cifar+50"):
		trans = [transforms.Resize((28, 28)),
			transforms.ToTensor(),
		]
		dataset_train = datasets.CIFAR10(data_dir, train=True, download=False,
				transform=transforms.Compose(trans))
		dataset_test = datasets.CIFAR10(data_dir, train=False, download=False,
				transform=transforms.Compose(trans))
		dataset = datasets.CIFAR100(data_dir, train=False, download=True,
									transform=transforms.Compose(trans))
		if dataset_name == 'cifar+10':
			class_list = get_cifar10plus_splits(cifar10_classes=dataset_test.classes, split=split)
			unknown_classes = get_cifar10plus_splits(cifar100_classes=dataset.classes, split=split)
		elif dataset_name == 'cifar+50':
			class_list = get_cifar50plus_splits(cifar10_classes=dataset.classes, split=split)
			unknown_classes = get_cifar50plus_splits(cifar100_classes=dataset.classes, split=split)

		dataset_test = DatasetSubset(dataset_test, class_list, return_real_classes=True)

		dataset = DatasetSubset(dataset, unknown_classes, return_real_classes=False, return_outlier_classes=True)
		dataset = MergeDatasets([dataset_test, dataset])
		dataset = DatasetSubset(dataset, class_list + [-1])

	# Tiny-ImageNet
	elif dataset_name == "tiny_imagenet":
		trans = [
			transforms.Resize((28, 28)),
			transforms.ToTensor(),
			GraImg2ColorImg()]
		dataset = TinyImagenetDataset(data_dir, train=False, download=True,
			transforms=transforms.Compose(trans))
		class_list, unknown_classes = get_tiny_imagenet_splits(dataset.labels_list, split)
		dataset.classes = list(range(200))
		dataset = DatasetSubset(dataset, class_list + unknown_classes)

	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, shuffle=False,
		num_workers=num_workers, pin_memory=pin_memory,
	)
	return data_loader

