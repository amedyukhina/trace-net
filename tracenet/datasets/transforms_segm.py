from torchvision import transforms


def get_train_transform_segm(patch_size=64):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(patch_size, pad_if_needed=True),
    ])


def get_valid_transform_segm(patch_size=64):
    return transforms.Compose([
        transforms.RandomCrop(patch_size, pad_if_needed=True),
    ])


def get_intensity_transform():
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])
