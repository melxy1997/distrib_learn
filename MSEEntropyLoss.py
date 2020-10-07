import torch
from torch import nn

class MSEEntropyLoss(nn.Module):
    def __init__(self):
        super(MSEEntropyLoss, self).__init__()
        self.MSE = nn.MSELoss()
    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        """
        Function that measures MSE of entropy value between target and output logits:
        example：
        MSEEntropyLoss(input, output) # input is the prediction value, and the output is the ground truth
        """
        assert p_target.shape == p_estimate.shape
        entropy_target = torch.sum(-p_target.mul(p_target.log()), dim=1)
        entropy_estimate = torch.sum(-p_estimate.mul(p_estimate.log()), dim=1)
        entropy_diff = self.MSE(entropy_estimate, entropy_target)
        return entropy_diff.mean()