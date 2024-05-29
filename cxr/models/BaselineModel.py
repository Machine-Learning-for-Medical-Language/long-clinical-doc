import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMortalityPredictor(nn.Module):
    def __init__(self, shape, num_filters=32, kernel_size=5, pool_size=10):

        super(BaselineMortalityPredictor, self).__init__()
        # in = embed_dim, out = num_filters, kernel = 5
        # default stride=1, padding=0, dilation=1
        stride = 1
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size)
        self.hout_conv = int(torch.floor( torch.tensor(1 + (shape[0] - (kernel_size-1) - 1) / stride)).item())
        self.wout_conv = int(torch.floor( torch.tensor(1 + (shape[1] - (kernel_size-1) - 1) / stride)).item())
        
        self.pool = nn.MaxPool2d(pool_size)
        pool_stride = pool_size
        self.hout_pool = int(torch.floor( torch.tensor(1 + (self.hout_conv - (pool_size-1) - 1) / pool_stride)).item())
        self.wout_pool = int(torch.floor( torch.tensor(1 + (self.wout_conv - (pool_size-1) - 1) / pool_stride)).item())
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv3 = nn.Conv2d(16, 32, 3)

        self.fc1 = nn.Linear(num_filters * self.hout_pool * self.wout_pool, 2)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        print("Model initialized with hidden layer containing %d nodes" % (self.fc1.in_features) )
    
    def forward(self, matrix):
        unpooled = F.relu(self.conv1(matrix))
        pooled = self.pool(unpooled)
        x = pooled.view(matrix.shape[0], -1)
        out = self.fc1(x)
        return out
