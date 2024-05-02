import matplotlib.pyplot as plt
import numpy as np

def display_image(image):
    img = image / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()