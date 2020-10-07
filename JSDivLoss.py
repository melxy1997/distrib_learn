import torch
import torch.nn.functional as F

def JSDivLoss(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    example：
    JS_div(input.log(), output) # input is the prediction value, and the output is the ground truth
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2