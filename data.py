from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Authorized domain = CIFAR-10, restricted domain = SVHN.
# Both are 32x32x3 with 10 classes, so no input/output reshaping is needed and
# the "different domain" signal is clean.
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


def _tf():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def cifar10_loaders(root="./data", batch_size=128, num_workers=2):
    tf = _tf()
    train = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True),
            DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers))


def svhn_loaders(root="./data", batch_size=128, num_workers=2):
    tf = _tf()
    train = datasets.SVHN(root, split="train", download=True, transform=tf)
    test = datasets.SVHN(root, split="test", download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, drop_last=True),
            DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers))


def infinite(loader):
    """Yield batches forever so the meta-loops can pull a batch at a time."""
    while True:
        for batch in loader:
            yield batch
