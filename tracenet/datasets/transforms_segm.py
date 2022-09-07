from torchvision import transforms


def get_train_transform_segm(patch_size=64):
    return transforms.Compose([
        transforms.RandomCrop(patch_size, pad_if_needed=True),
    ])


def get_valid_transform_segm(patch_size=64):
    return transforms.Compose([
        transforms.RandomCrop(patch_size, pad_if_needed=True),
    ])
