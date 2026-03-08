from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class MimicDataset(Dataset):
    def __init__(
        self,
        split='train',
        train_txt='/fast/yangz16/outputs/x-win/train_mimic.txt',
        test_txt='/fast/yangz16/outputs/x-win/test_mimic.txt',
        transform=None,
        return_path=False,
    ):
        super().__init__()
        assert split in ['train', 'val', 'test'], "split must be one of: 'train', 'val', 'test'"
        self.split = split
        self.train_txt = Path(train_txt)
        self.test_txt = Path(test_txt)
        self.transform = transform
        self.return_path = return_path
        self.image_paths = self.get_image_paths()

    def _read_txt(self, txt_path: Path):
        if not txt_path.exists():
            raise FileNotFoundError(f"List file not found: {txt_path}")
        with txt_path.open('r') as f:
            return [line.strip() for line in f if line.strip()]

    def get_image_paths(self):
        list_path = self.train_txt if self.split == 'train' else self.test_txt
        return self._read_txt(list_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.return_path:
            return img, path
        return img
