
# dataset labels
labels = dict(
    cifar10=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    cifar100=['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']
)

# dataset orig img sizes
img_sizes = dict(
    cifar10=(64, 64),
    cifar100=(64, 64)
)

# dataset configs
dataset_cfg = dict(
    cifar10=dict(
        name='cifar10',
        labels=labels['cifar10'],
        num_cls=len(labels['cifar10']),
        input_dim=3,
        path='../data/CIFAR/',
        img_sizes=img_sizes['cifar10']
    ),
    cifar100=dict(
        name='cifar100',
        labels=labels['cifar100'],
        num_cls=len(labels['cifar100']),
        input_dim=3,
        path='../data/CIFAR/',
        img_sizes=img_sizes['cifar100']
    )
)

# paths
model_dir = './trained_models/'