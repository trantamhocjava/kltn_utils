import random

import matplotlib.colors as mcolors
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


def plot_bar_chart(
    x_values,
    y_values,
    horizontal=False,
    title="",
    xlabel="",
    ylabel="",
    figsize=(10, 5),
):
    plt.figure(figsize=figsize)

    if horizontal:
        plt.barh(x_values, y_values)
        plt.gca().invert_yaxis()
        plt.xlabel(ylabel)
        plt.ylabel(xlabel)
        plt.xlim(left=0)
    else:
        plt.bar(x_values, y_values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.ylim(bottom=0)

    plt.title(title)
    plt.tight_layout()
    plt.show()


def style_same_values_with_same_color(df, skip_cols=("criteria",)):
    """
    Tô cùng 1 màu cho các cell có cùng giá trị.
    Ví dụ: mọi cell chứa 'microsoft/git-large-r-textcaps' sẽ cùng 1 màu.
    skip_cols: các cột cần bỏ qua
    """

    # Lấy các giá trị cần tô màu
    values = []
    for col in df.columns:
        if col in skip_cols:
            continue
        values.extend(df[col].dropna().astype(str).unique())

    values = sorted(set(values))

    # Tạo bảng màu
    color_names = list(mcolors.TABLEAU_COLORS.values()) + list(
        mcolors.CSS4_COLORS.values()
    )

    value_to_color = {
        value: color_names[i % len(color_names)] for i, value in enumerate(values)
    }

    def color_cell(value):
        value = str(value)

        if value in value_to_color:
            return f"background-color: {value_to_color[value]}; color: white;"

        return ""

    return df.style.map(color_cell)
