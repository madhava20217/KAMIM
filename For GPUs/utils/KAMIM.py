import torch

@torch.no_grad()
def per_patch_importance(batch, patch_size, temperature):
    '''Function to get per-patch importance given a feature descriptor for each pixel in the image
    
    Arguments:
    1. batch
    2. num_patches: number of patches along each side of the image
    3. size_patch: number of pixels along each side of the patch
    4. temperature: a scaling value for relative importance
    
    Returns:
    1. per-patch importance for each patch'''
    n_batch = batch.size(0)
    features = torch.nn.functional.conv2d(batch,
                                  weight = torch.ones(1, 1, patch_size,patch_size, device = batch.device)/(patch_size**2), 
                                  bias = torch.zeros(1, device = batch.device),
                                  stride = (patch_size, patch_size)).squeeze(1)
    patch_weights = torch.exp(torch.clip(features/temperature, min = -20, max = 20))
    patch_weights = patch_weights/patch_weights.view(n_batch, -1).min(axis = 1).values.view(-1,1,1)
    return patch_weights


class ModifiedPixelShuffle(torch.nn.PixelShuffle):
        r"""Rearrange elements in a tensor according to an upscaling factor.

    Rearranges elements in a tensor of shape :math:`(*, C \times r^2, H, W)`
    to a tensor of shape :math:`(*, C, H \times r, W \times r)`, where r is an upscale factor.

    This is useful for implementing efficient sub-pixel convolution
    with a stride of :math:`1/r`.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.

    Args:
        upscale_factor (int): factor to increase spatial resolution by

    Shape:
        - Input: :math:`(*, C_{in}, H_{in}, W_{in})`, where * is zero or more batch dimensions
        - Output: :math:`(*, C_{out}, H_{out}, W_{out})`, where

    .. math::
        C_{out} = C_{in} \div \text{upscale\_factor}^2

    .. math::
        H_{out} = H_{in} \times \text{upscale\_factor}

    .. math::
        W_{out} = W_{in} \times \text{upscale\_factor}

    Examples::

        >>> pixel_shuffle = nn.PixelShuffle(3)
        >>> input = torch.randn(1, 9, 4, 4)
        >>> output = pixel_shuffle(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])

    .. _Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network:
        https://arxiv.org/abs/1609.05158
        
        MODIFIED FOR TPUs (bug present): https://github.com/pytorch/xla/issues/5886
    """

        def __init__(self, upscale_factor: int) -> None:
            super().__init__(upscale_factor)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.pixel_shuffle_TPU(input, self.upscale_factor)
        
        
        def pixel_shuffle_TPU(self, tensor: torch.Tensor, upscale_factor: int):
            '''Modified implementation of PixelShuffle for TPUs given torch_xla issue #5886 that returns all zeros'''
            b, c, h, w = tensor.shape
            oc = c//upscale_factor**2
            oh = h*upscale_factor
            ow = w*upscale_factor
                                
            return (((
                        tensor
                        ).view(b, oc, upscale_factor, upscale_factor, h, w) 
                    ).permute(0, 1, 4, 2, 5, 3)           # b, oc, h, up1, w, up2
                ).reshape(b, oc, oh, ow).clone()