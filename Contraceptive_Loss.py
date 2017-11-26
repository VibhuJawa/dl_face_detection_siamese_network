import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self,output_1 , output_2, label):
        distance = F.pairwise_distance(output_1, output_2)
        loss_1 = (label) * torch.pow(distance, 2)
        loss_2 = (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0), 2)
        loss = (loss_1 + loss_2).mean()
        return loss
