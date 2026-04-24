import random

import matplotlib.pyplot as plt
from PIL import Image


def show_img(
    image_path,
    width=8,
    height=6,
):
    img = Image.open(image_path)

    # độ phân giải ảnh
    img_width, img_height = img.size

    # định dạng file
    file_format = img.format

    plt.figure(figsize=(width, height))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print(f"resolution: {img_width} x {img_height}")
    print(f"file format: {file_format}")


def random_sublist(src_list, num, replace=False):
    if replace:
        sub_list = random.choices(src_list, k=num)
    else:
        sub_list = random.sample(src_list, num)

    return sub_list


def show_images(image_paths, m, n, figsize=(12, 8)):
    fig, axes = plt.subplots(m, n, figsize=figsize)

    # Trường hợp m*n = 1 thì axes không phải mảng
    if m == 1 and n == 1:
        axes = [[axes]]
    elif m == 1:
        axes = [axes]
    elif n == 1:
        axes = [[ax] for ax in axes]

    for i in range(len(image_paths)):
        row = i // n
        col = i % n
        ax = axes[row][col]

        img = Image.open(image_paths[i])
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
