import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes

FASHION_MNIST_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def show_training_image(dataset):
    plt.imshow(dataset[0][0].squeeze(), cmap='gray')
    plt.title('First training image')
    plt.show()

def show_test_image(dataset):
    plt.imshow(dataset[0][0].squeeze(), cmap='gray')
    plt.title('First test image')
    plt.show()

def show_more_image(dataset, num_image=24):
    cols = rows = int(num_image ** 0.5)
    
    axes: np.ndarray
    
    figur,axes = plt.subplots(rows,cols)

    for ax, (image,label_number) in zip(axes.flat,dataset):
        ax:Axes
        image:torch.Tensor

        ax.imshow(image.squeeze(),cmap="grey")
        ax.set_title(f"Label: {FASHION_MNIST_LABELS[label_number]} {label_number}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_label_one(dataset):
    for image, label in dataset:
        if label == 0:  # 1 is Trouser
            plt.imshow(image.squeeze(), cmap="gray")
            plt.title(f"Label 1: Trouser")
            plt.axis('off')
            plt.show()
            break  