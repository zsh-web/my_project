import torch
from torchvision import datasets, transforms
import numpy as np
import cv2
from scipy.signal import wiener

# --------- 数据选取函数 ---------
def select_data(dataset, target_classes, train_per_class, test_per_class):
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    train_counts = {c: 0 for c in target_classes}
    test_counts = {c: 0 for c in target_classes}

    for img, label in dataset:
        lbl = label.item()
        if lbl in target_classes:
            img_np = img.squeeze().numpy()
            if train_counts[lbl] < train_per_class:
                train_images.append(img_np)
                train_labels.append(lbl)
                train_counts[lbl] += 1
            elif test_counts[lbl] < test_per_class:
                test_images.append(img_np)
                test_labels.append(lbl)
                test_counts[lbl] += 1

        if all(train_counts[c] >= train_per_class for c in target_classes) and \
           all(test_counts[c] >= test_per_class for c in target_classes):
            break

    return (np.array(train_images), np.array(train_labels),
            np.array(test_images), np.array(test_labels))

# --------- 高斯金字塔降维 ---------
def apply_gaussian_pyramid(images, levels=1):
    downsampled = []
    for img in images:
        img_pyr = img.copy()
        for _ in range(levels):
            img_pyr = cv2.pyrDown(img_pyr)
        downsampled.append(img_pyr)
    return downsampled

# --------- 添加高斯噪声 ---------
def add_gaussian_noise(images, mean=0, std=0.05):
    noisy_images = []
    for img in images:
        noise = np.random.normal(mean, std, img.shape)
        noisy = img + noise
        noisy = np.clip(noisy, 0, 1)
        noisy_images.append(noisy.astype(np.float32))
    return noisy_images

# --------- 迭代贝叶斯去噪（迭代Wiener滤波） ---------
def iterative_wiener_denoise(images, iterations=5, mysize=3):
    denoised = []
    for img in images:
        est = img.copy()
        for _ in range(iterations):
            est = wiener(est, mysize=mysize)
        est = np.clip(est, 0, 1)
        denoised.append(est.astype(np.float32))
    return denoised

# --------- 主流程 ---------
transform = transforms.ToTensor()

# 下载合并数据集
mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
mnist_all = torch.utils.data.ConcatDataset([mnist_train, mnist_test])

fashion_train = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
fashion_test = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
fashion_all = torch.utils.data.ConcatDataset([fashion_train, fashion_test])

target_classes = [3, 4, 5]
train_per_class = 400
test_per_class = 2000

# 选数据
mnist_train_imgs, mnist_train_labels, mnist_test_imgs, mnist_test_labels = select_data(
    mnist_all, target_classes, train_per_class, test_per_class)
fashion_train_imgs, fashion_train_labels, fashion_test_imgs, fashion_test_labels = select_data(
    fashion_all, target_classes, train_per_class, test_per_class)

# 降维
mnist_train_pyr = apply_gaussian_pyramid(mnist_train_imgs, levels=1)
mnist_test_pyr = apply_gaussian_pyramid(mnist_test_imgs, levels=1)
fashion_train_pyr = apply_gaussian_pyramid(fashion_train_imgs, levels=1)
fashion_test_pyr = apply_gaussian_pyramid(fashion_test_imgs, levels=1)

# 添加噪声
mnist_train_noisy = add_gaussian_noise(mnist_train_pyr)
mnist_test_noisy = add_gaussian_noise(mnist_test_pyr)
fashion_train_noisy = add_gaussian_noise(fashion_train_pyr)
fashion_test_noisy = add_gaussian_noise(fashion_test_pyr)

# 迭代贝叶斯去噪
mnist_train_denoised = iterative_wiener_denoise(mnist_train_noisy)
mnist_test_denoised = iterative_wiener_denoise(mnist_test_noisy)
fashion_train_denoised = iterative_wiener_denoise(fashion_train_noisy)
fashion_test_denoised = iterative_wiener_denoise(fashion_test_noisy)

