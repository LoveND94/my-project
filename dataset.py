import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class CRLMDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        self.labels = [1 if 'pos' in f else 0 for f in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = img.resize((224, 224))
        tensor = torch.FloatTensor(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).float()).view(3, 224, 224) / 255.
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label
