# dataset.py
import torch, pandas as pd, cv2
from torch.utils.data import Dataset
from torchvision import transforms

class PairFoodDataset(Dataset):
    def __init__(self, csv_path, img_size=224, augment=False):
        self.df = pd.read_csv(csv_path)
        self.augment = augment

        self.base_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])
        # Augment ringan (hanya untuk train)
        self.aug_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05)),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std =[0.229,0.224,0.225]),
        ])

    def _read_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        im_before = self._read_img(row['before_path'])
        im_after  = self._read_img(row['after_path'])
        tf = self.aug_tf if self.augment else self.base_tf
        x_before = tf(im_before)
        x_after  = tf(im_after)
        # label dari 1..7 -> 0..6
        y = int(row['label']) - 1
        return x_before, x_after, y
