from setuptools import find_packages, setup

setup(
    name="kltn_utils",  # Tên thư viện của bạn
    version="0.1.0",  # Phiên bản thư viện
    packages=find_packages(),  # Tìm các package trong thư mục hiện tại
    install_requires=[
        "git+https://github.com/openai/CLIP.git"
    ],  # Các thư viện phụ thuộc (nếu có)
)
