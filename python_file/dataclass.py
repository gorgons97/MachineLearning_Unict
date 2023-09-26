import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

class StreetSignDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        
        # Carica le landmarks come array NumPy e assicurati che abbiano sempre dimensioni consistenti
        landmarks = np.array(str(self.landmarks_frame.iloc[idx, 1]).split(), dtype=np.float32)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class StreetSignTest(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)

        top_w_str = str(self.landmarks_frame.iloc[idx, 1]).strip()
        top_h_str = str(self.landmarks_frame.iloc[idx, 2]).strip()
        bottom_w_str = str(self.landmarks_frame.iloc[idx, 3]).strip()
        bottom_h_str = str(self.landmarks_frame.iloc[idx, 4]).strip()

        top_w = int(top_w_str)
        top_h = int(top_h_str)
        bottom_w = int(bottom_w_str)
        bottom_h = int(bottom_h_str)


        # Suddivide la stringa in una lista di valori numerici utilizzando gli spazi come separatore
        top_w = map(int, top_w_str.split())
        top_h = map(int, top_h_str.split())
        bottom_w = map(int, bottom_w_str.split())
        bottom_h = map(int, bottom_h_str.split())


        # Usare direttamente i valori delle coordinate
                
        image = image[top_h:bottom_h, top_w:bottom_w]
        
        # Carica le landmarks come array NumPy e assicurati che abbiano sempre dimensioni consistenti
        landmarks = np.array(str(self.landmarks_frame.iloc[idx, 5]).split(), dtype=np.float32)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'landmarks': landmarks}
    
class Crop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, top_w, top_h, bottom_w, bottom_h):
        assert isinstance(top_h, top_w, bottom_h, bottom_w, (int, tuple))
        if isinstance(top_h, top_w, bottom_h, bottom_w, int):
            self.top_h = top_h
            self.top_w = top_w
            self.bottom_h = bottom_h
            self.bottom_w = bottom_w
            self.height = abs(self.top_h - self.bottom_h)
            self.width = abs(self.top_w - self.bottom_w)
            if self.height < 32:
                self.height = 32
            if self.width < 32:
                self.width = 32

    def __call__(self, sample):
        image = sample['image']
        image = image[top: self.top_h + self.height,
                      left: self.top_w + self.width]

        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        landmarks = np.array(landmarks, dtype=np.float32)  # Converti landmarks in un array NumPy
        
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
        sample_batched['image'], sample_batched['landmarks']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        print(f"Landmarks shape for sample {i}: {landmarks_batch[i].shape}")
        landmarks = landmarks_batch[i].view(-1, 2)  # Reshape per renderli bidimensionali
        print(f"Reshaped landmarks shape for sample {i}: {landmarks.shape}")
        plt.scatter(landmarks[:, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks[:, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')

    plt.title('Batch from dataloader')