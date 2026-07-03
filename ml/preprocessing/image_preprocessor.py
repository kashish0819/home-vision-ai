import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():

    return A.Compose([

        A.Resize(224, 224),

        A.HorizontalFlip(p=0.5),

        A.VerticalFlip(p=0.2),

        A.RandomBrightnessContrast(p=0.3),

        A.Rotate(limit=15, p=0.3),

        A.Normalize(),

        ToTensorV2()

    ])


def get_valid_transforms():

    return A.Compose([

        A.Resize(224, 224),

        A.Normalize(),

        ToTensorV2()

    ])