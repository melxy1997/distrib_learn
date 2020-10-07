import torch
from torch import nn
import torch.nn.functional as F

class JSDivLoss(nn.Module):
    def __init__(self):
        super(JSDivLoss, self).__init__()
    def forward(self, p_output, q_output, get_sigmoid=True):
        """
        Function that measures JS divergence between target and output logits:
        example：
        JS_div(input.log(), output) # input is the prediction value, and the output is the ground truth
        """
        KLDivLoss = nn.KLDivLoss(reduction='batchmean')
        if get_sigmoid:
            p_output = F.sigmoid(p_output)
            q_output = F.sigmoid(q_output)
        log_mean_output = ((p_output + q_output )/2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2
