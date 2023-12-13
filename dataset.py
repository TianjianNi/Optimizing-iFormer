from torch.utils.data import Dataset
from torchvision import datasets
import os


class Customized_CIFAR10_Dataset(Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(os.path.join(self.root, 'train')) if
                               os.path.isdir(os.path.join(self.root, 'train', d))])
        self.data = []
        for class_id, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root, 'train', class_name)
            for filename in os.listdir(class_path):
                if filename.lower().endswith('.png') and filename.startswith('img'):
                    image_path = os.path.join(class_path, filename)
                    # Extract the image number from the filename
                    img_number = int(filename[3:-4])  # Assuming 'img' prefix and '.png' suffix
                    self.data.append((image_path, class_id, img_number))

        # Sort the data based on the image number
        self.data.sort(key=lambda x: x[2])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label, img_number = self.data[index]
        image = datasets.folder.default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label