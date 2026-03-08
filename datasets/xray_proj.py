import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


def get_path_info(path):
    dirname = os.path.dirname(path)
    filename = os.path.basename(path).rstrip('.jpg')
    name, pos = filename.rsplit('_', 1)
    return dirname, name, int(pos)


class XrayProjDataset(Dataset):
    def __init__(self, root_dir, split='train', step=15, transform=None, ctx_only=False):
        super().__init__()
        assert split in ['train', 'val', 'test'], "split must be 'train' or 'test'"
        self.split = split
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self.get_image_paths()
        self.action_space = np.arange(-90, 91, step)

        self.ctx_only = ctx_only

    def get_image_paths(self):
        with open(self.root_dir, 'r') as f:
            content = f.read()
        paths = content.splitlines()

        # return paths
        paths_front = []
        for p in paths:
            _, _, pos = get_path_info(p)
            if pos == 0:
                paths_front.append(p)

        return paths_front

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load context image
        path_ctx = self.image_paths[idx]
        dirname, name, _ = get_path_info(path_ctx)
        img_ctx = Image.open(path_ctx).convert('RGB')

        # Load target image
        action = np.random.choice(self.action_space)
        path_tgt = os.path.join(dirname, name) + f'_{int(action)}.jpg'
        img_tgt = Image.open(path_tgt).convert('RGB')

        # Apply transforms if provided
        if self.transform:
            img_ctx = self.transform(img_ctx)
            img_tgt = self.transform(img_tgt)

        action = np.deg2rad(action)
        action = torch.tensor([action], dtype=torch.float32)

        if self.ctx_only:
            return img_ctx, path_ctx
        else:
            return img_ctx, img_tgt, action


class ProjCollator(object):
    def __init__(self, step=15, num_samples=5, transform=None):
        self.step = step
        self.num_samples = num_samples
        self.transform = transform
        self.action_space = np.arange(-90, 91, self.step)

    def __call__(self, batch):
        B = len(batch)

        batch = torch.utils.data.default_collate(batch)  # tensor (B, C, H, W), tuple 24

        collated_tgt, collated_actions = [], []
        ctx_paths = batch[1]
        for i in range(B):
            dirname, name, _ = get_path_info(ctx_paths[i])

            actions = np.random.choice(self.action_space, size=self.num_samples, replace=False)
            tgt_paths = [os.path.join(dirname, name) + f'_{int(a)}.jpg' for a in actions]

            # Load and transform context images
            tgt_imgs = [self.transform(Image.open(_).convert('RGB')).unsqueeze(0) for _ in tgt_paths]
            tgt_imgs = torch.cat(tgt_imgs, dim=0)  # (N, 3, H, W)

            actions_rad = np.deg2rad(actions)
            actions_rad = torch.tensor(actions_rad, dtype=torch.float32).unsqueeze(1)  # (N, 1)

            collated_tgt.append(tgt_imgs)
            collated_actions.append(actions_rad)

        collated_tgt = torch.utils.data.default_collate(collated_tgt)  # (B, N, 1)
        collated_actions = torch.utils.data.default_collate(collated_actions)  # (B, N, 3, H, W)

        b, n, c, h, w = collated_tgt.shape
        collated_tgt = collated_tgt.view(b*n, c, h, w)
        collated_actions = collated_actions.view(b*n, 1)

        return batch, collated_tgt, collated_actions