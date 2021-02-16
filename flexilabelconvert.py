import os, random

MNIST_SPLITS = [[3, 6, 7, 8],
                [1, 2, 4, 6],
                [2, 3, 4, 9],
                [0, 1, 2, 6],
                [4, 5, 6, 9]]
cifar100_splits = [['fox', 'lobster', 'porcupine', 'elephant', 'woman', 'lizard', 'turtle', 'crab', 'rabbit', 'snail'],
                   ['wolf', 'whale', 'beetle', 'woman', 'raccoon', 'turtle', 'boy', 'skunk', 'worm', 'snail'],
                   ['fox', 'porcupine', 'whale', 'camel', 'baby', 'flatfish', 'tiger', 'boy', 'cockroach', 'turtle'],
                   ['leopard', 'possum', 'lobster', 'aquarium_fish', 'porcupine', 'raccoon', 'ray', 'lizard', 'boy', 'kangaroo'],
                   ['leopard', 'snake', 'bear', 'camel', 'chimpanzee', 'man', 'woman', 'raccoon', 'seal', 'cattle']]
cifar100_vehicles = ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
def get_mnist_svhn_cifar10_splits(classes, split='split0'):
    known_classes = []
    unknown_classes = MNIST_SPLITS[int(split[-1])]
    for item in range(10):
        if item not in unknown_classes:
            known_classes += [item]
    return known_classes, unknown_classes

def get_cifar10plus_splits(cifar10_classes=None, cifar100_classes=None, split='split0'):
    if cifar10_classes:
        known_classes = ['airplane', 'automobile', 'ship', 'truck']
        known_labels = []
        unknown_labels = []
        for item in known_classes:
            known_labels += [cifar10_classes.index(item)]
        for item in range(10):
            if item not in known_labels:
                unknown_labels += [item]
        _labels = known_labels
    if cifar100_classes:
        cifar100_unknown_labels = []
        for item in cifar100_splits[int(split[-1])]:
            cifar100_unknown_labels += [cifar100_classes.index(item)]
        _labels = cifar100_unknown_labels
    return sorted(_labels)

def get_cifar50plus_splits(cifar10_classes=None, cifar100_classes=None, split='split0'):
    if cifar10_classes:
        known_classes = ['airplane', 'automobile', 'ship', 'truck']
        known_labels = []
        unknown_labels = []
        for item in known_classes:
            known_labels += [cifar10_classes.index(item)]
        for item in range(10):
            if item not in known_labels:
                unknown_labels += [item]
        _labels = known_labels
    if cifar100_classes:
        idx = int(split[-1])
        temp_classes = cifar100_classes[:]
        random.seed(42)
        for i in range(idx+1):
            random.shuffle(temp_classes)
        count = 0
        unknown_labels = []
        for item in temp_classes:
            if item not in cifar100_vehicles:
                unknown_labels += [cifar100_classes.index(item)]
                count += 1
                if count >= 50:
                    break
        _labels = sorted(unknown_labels)
    return sorted(_labels)

def get_tiny_imagenet_splits(classes, split='split0'):
    idx = int(split[-1])
    temp_classes = classes[:]
    random.seed(42)
    for i in range(idx+1):
        random.shuffle(temp_classes)
    known_labels = []
    known_classes = temp_classes[:20]
    for item in known_classes:
        known_labels += [classes.index(item)]
    unknown_labels = []
    for item in classes:
        if item not in known_classes:
            unknown_labels += [classes.index(item)]

    if idx== 5:
        known_labels = [i for i in range(20)]
        unknown_labels = [i for i in range(20, 200)]
    return sorted(known_labels), sorted(unknown_labels)