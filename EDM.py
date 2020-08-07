import torch

class EDM(nn.Module):
    r"""The Earth Move Distance from the
    `"The Earth Mover’s distance is the Mallows distance: 
some insights from statistics. (ICCV 2001)" `_ paper
    .. math::
        \mathbf{X}^{\prime} = \sum_{k=1}^{K} \mathbf{Z}^{(k)} \cdot
        \mathbf{\Theta}^{(k)}
    where :math:`\mathbf{Z}^{(k)}` is computed recursively by
    .. math::
        \mathbf{Z}^{(1)} &= \mathbf{X}
        \mathbf{Z}^{(2)} &= \mathbf{\hat{L}} \cdot \mathbf{X}
        \mathbf{Z}^{(k)} &= 2 \cdot \mathbf{\hat{L}} \cdot
        \mathbf{Z}^{(k-1)} - \mathbf{Z}^{(k-2)}
    and :math:`\mathbf{\hat{L}}` denotes the scaled and normalized Laplacian
    :math:`\frac{2\mathbf{L}}{\lambda_{\max}} - \mathbf{I}`.
    """
    def __init__(self):
        super(EDM, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        # cdf for values [1, 2, ..., 10]
        cdf_target = torch.cumsum(p_target, dim=1)
        # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_estimate, dim=1)
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2)))
        return samplewise_emd.mean()