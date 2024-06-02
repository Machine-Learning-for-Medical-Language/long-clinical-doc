import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import imageio

class MultiChannelMortalityPredictor(nn.Module):
    def __init__(self, shape, num_filters=32, kernel_size=(1,5,5), pool_size=10):

        super(MultiChannelMortalityPredictor, self).__init__()
        # in = embed_dim, out = num_filters, kernel = 5
        # default stride=1, padding=0, dilation=1
        stride = 1
        # use a 3d convolution because there will be several images per instance, but variable numbers
        self.conv1 = nn.Conv3d(1, num_filters, kernel_size)
        # self.dout_conv = int(torch.floor( torch.tensor(1 + (shape[0] - (kernel_size[0]-1) - 1) / stride)).item())
        self.hout_conv = int(torch.floor( torch.tensor(1 + (shape[0] - (kernel_size[1]-1) - 1) / stride)).item())
        self.wout_conv = int(torch.floor( torch.tensor(1 + (shape[1] - (kernel_size[2]-1) - 1) / stride)).item())
        
        self.pool_size = pool_size
        pool_stride = pool_size
        # self.dout_pool = int(torch.floor( torch.tensor(1 + (self.dout_conv - (pool_size-1) - 1) / pool_stride)).item())
        self.hout_pool = int(torch.floor( torch.tensor(1 + (self.hout_conv - (pool_size-1) - 1) / pool_stride)).item())
        self.wout_pool = int(torch.floor( torch.tensor(1 + (self.wout_conv - (pool_size-1) - 1) / pool_stride)).item())
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(num_filters * self.hout_pool * self.wout_pool, 2)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        print("Model initialized with hidden layer containing %d nodes" % (self.fc1.in_features) )
    
    def forward(self, matrix):
        channels = matrix.shape[2]
        unpooled = F.relu(self.conv1(matrix))
        # pooled = self.pool(unpooled)
        pooled = F.max_pool3d(unpooled, (channels, self.pool_size, self.pool_size))

        x = pooled.view(matrix.shape[0], -1)
        out = self.fc1(x)
        return out



class VariableLengthImageDataset(Dataset):
    def __init__(self, data_paths, labels, transform=None):
        """
        Args:
            data_paths (list of lists): List where each element is a list of file paths to images for a single instance.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        image_paths = self.data_paths[idx]
        # images = [Image.open(image_path) for image_path in image_paths]
        images = [imageio.imread(img_path, mode='F') for img_path in image_paths]
        if self.transform:
            images = [self.transform(image) for image in images]
        
        return images, self.labels[idx]


# Custom collate function to handle variable-length batches
def collate_fn(batch):
    images_batch, labels_batch = zip(*batch)
    max_length = max(len(image_list) for image_list in images_batch)
    padded_batch = []

    # Each instance has a list of images associated with it of variable size. Inside each batch, pad
    # out the lists with empty images then turn into a tensor
    for image_list in images_batch:
        padded_list = image_list + [torch.zeros_like(image_list[0])] * (max_length - len(image_list))
        padded_batch.append(torch.stack(padded_list))
    
    # squeeze the dimension corresponding to the grayscale channel, make sure to specify dim=2 otherwise
    # it could squeeze the batch dimension if batch_size=1
    return torch.squeeze(torch.stack(padded_batch), dim=2), torch.tensor(labels_batch)
