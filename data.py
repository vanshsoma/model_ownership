from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Authorized domain = CIFAR-10. Restricted domains are any of the others.
# Everything is mapped to 3x32x32 with 10 classes, so a single ResNet-18 backbone
# and a 10-way Head work for every domain — which is what lets us train the
# suppression on a BASKET of domains and evaluate on a HELD-OUT one.
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)

# name -> (torchvision class, uses split= instead of train=, is_grayscale)
_DOMAINS = {
    "cifar10": (datasets.CIFAR10,      False, False),
    "svhn":    (datasets.SVHN,         True,  False),
    "mnist":   (datasets.MNIST,        False, True),
    "fashion": (datasets.FashionMNIST, False, True),
    "kmnist":  (datasets.KMNIST,       False, True),
    "usps":    (datasets.USPS,         False, True),
}


def _tf(gray):
    ops = []
    if gray:
        ops.append(transforms.Grayscale(num_output_channels=3))
    ops += [transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD)]
    return transforms.Compose(ops)


def domain_loaders(name, root="./data", batch_size=128, num_workers=2, limit=None):
    """Train/test loaders for any registered 10-class domain, all as 3x32x32.

    `limit` restricts the training set (few-shot regime for the attacker).
    """
    cls, uses_split, gray = _DOMAINS[name]
    tf = _tf(gray)
    if uses_split:
        train = cls(root, split="train", download=True, transform=tf)
        test = cls(root, split="test", download=True, transform=tf)
    else:
        train = cls(root, train=True, download=True, transform=tf)
        test = cls(root, train=False, download=True, transform=tf)
    if limit is not None:
        train = Subset(train, list(range(min(limit, len(train)))))
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=(limit is None)),
            DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers))


# Backward-compatible wrappers (older commands still work).
def cifar10_loaders(root="./data", batch_size=128, num_workers=2):
    return domain_loaders("cifar10", root, batch_size, num_workers)


def svhn_loaders(root="./data", batch_size=128, num_workers=2, limit=None):
    return domain_loaders("svhn", root, batch_size, num_workers, limit)


def infinite(loader):
    """Yield batches forever so the meta-loops can pull a batch at a time."""
    while True:
        for batch in loader:
            yield batch
