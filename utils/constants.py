
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

# Dataset Config Class
class DatasetConfig():

    def __init__(self, name, input_dim):
        self.name = name
        self.labels = labels[self.name]
        self.num_cls = len(labels[self.name])
        self.input_dim = 3
        self.path = '/project/panda/shared/' + self.name
        self.img_size = img_sizes[self.name]

    def dictionary(self):
        return dict(
            name=self.name,
            labels=self.labels,
            num_cls=self.num_cls,
            input_dim=self.input_dim,
            path=self.path,
            img_size=self.img_size
        )

# dataset configs
dataset_cfg = dict(
    cifar10=DatasetConfig(
        name='cifar10',
        input_dim=3,
    ).dictionary(),
    cifar100=DatasetConfig(
        name='cifar100',
        input_dim=3,
    ).dictionary()
)

# paths
model_dir = './trained_models/'