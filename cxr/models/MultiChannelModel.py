import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import imageio
from tqdm import tqdm
import h5py
import torchvision.models as models

class MultiChannelMortalityPredictor(nn.Module):
    def __init__(self, shape, embed_dim = 512, num_heads=16):
        super(MultiChannelMortalityPredictor, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Identity()
        self.multi_head_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.Q = torch.rand(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 2)

    def forward(self, batch_input):
        if not self.Q.device == batch_input.device:
            self.Q = self.Q.to(batch_input.device)

        # multiply out grayscale channels to RGB for resnet encoder
        rgb_matrix = torch.repeat_interleave(batch_input, 3, dim=1)
        # how many images are in this instance
        channels = rgb_matrix.shape[2]
        reps = torch.stack([self.encoder(rgb_matrix[:,:,i,:,:]) for i in range(channels)])
        
        attn_output, attn_weights = self.multi_head_attention(
            # Need to multiply out the query vector so there's one for each batch, then unsqueeze again for dim=1 output 
            torch.repeat_interleave(self.Q.unsqueeze(dim=0), batch_input.shape[0], dim=0).unsqueeze(dim=0),
            reps,
            reps
        )
        # squeeze the collapsed attention dimension but keep the batch dimension (in case batch size=1)
        out = self.fc1(attn_output).squeeze(dim=0)
        return out



class VariableLengthImageDataset(Dataset):
    def __init__(self, hdf5_fn):
        """
        Args:
        """
        self.fp = h5py.File(hdf5_fn, "r")
        self.labels = []
        for idx in range(len(self)):
            self.labels.append(self.fp['%d/label' % (idx)][()])

    # This weird [()] syntax is just how you retrieve scalars in hdf5
    def __len__(self):
        return self.fp['/len/'][()]

    def __getitem__(self, idx):
        return torch.tensor(self.fp['%d/data' % (idx)]), self.labels[idx]

    def get_metadata(self, idx):
        return self.fp['%d/hadm' % (idx)][()].decode()
    
    def get_id(self, idx):
        return self.fp['%d/id' % (idx)][()].decode()

# Custom collate function to handle variable-length batches
def collate_fn(batch):
    images_batch, labels_batch = zip(*batch)
    max_length = max(len(image_list) for image_list in images_batch)
    padded_batch = []

    # Each instance has a list of images associated with it of variable size. Inside each batch, pad
    # out the lists with empty images then turn into a tensor
    for image_list in images_batch:
        # padded_list = image_list + [torch.zeros_like(image_list[0])] * (max_length - len(image_list))
        padding = torch.zeros( (max_length-image_list.shape[0], 1, image_list.shape[2], image_list.shape[3] ) )
        padded_batch.append(torch.cat([image_list, padding]))
    
    # squeeze the dimension corresponding to the grayscale channel, make sure to specify dim=2 otherwise
    # it could squeeze the batch dimension if batch_size=1
    return torch.squeeze(torch.stack(padded_batch), dim=2), torch.tensor(labels_batch)
