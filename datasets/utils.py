from torchvision import transforms


def get_transform(image_size=224, crop_scale=(0.75, 1.0), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        # Resize so the shorter side == image_size
        transforms.RandomResizedCrop((image_size,)*2, scale=crop_scale),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
        # Convert PIL image to [0,1] tensor C×H×W
        transforms.ToTensor(),
        # Normalize tensor with mean/std
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transform(image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        # Resize so the shorter side == image_size
        transforms.Resize((image_size,)*2),
        # Convert PIL image to [0,1] tensor C×H×W
        transforms.ToTensor(),
        # Normalize tensor with mean/std
        transforms.Normalize(mean=mean, std=std),
    ])