import torchvision.models as models
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class VitMortalityPredictor(nn.Module):
    def __init__(self, shape, num_filters=64, kernel_size=5, pool_size=50):

        super(VitMortalityPredictor, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.encoder = AutoModel.from_pretrained('facebook/dinov2-small')

    def forward(self, matrix):
        output = self.encoder(matrix)
        logits = self.fc(output['pooler_output'])

        return logits
