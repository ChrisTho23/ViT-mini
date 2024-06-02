import matplotlib.pyplot as plt
import numpy as np
from torch import nn

def display_image(image, title=None):
    img = image / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()

class Depatchify(nn.Module):
    def __init__(self, patch_size, image_size):
        super().__init__()
        self.patch_size = patch_size
        self.image_size = image_size
        self.fold = nn.Fold(output_size=image_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, num_patches, patch_dim = x.shape # patch_dim = C * patch_size * patch_size
        x = x.permute(0, 2, 1) # B, patch_dim, num_patches
        x = self.fold(x) # B, C, H, W
        return x