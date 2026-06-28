from setuptools import find_packages, setup

setup(
    name="kltn_utils",  # Tên thư viện của bạn
    version="0.1.104",  # Phiên bản thư viện
    packages=find_packages(),  # Tìm các package trong thư mục hiện tại
    install_requires=[
        "transformers",
        (
            "hi-ml-multimodal @ "
            "git+https://github.com/microsoft/hi-ml.git"
            "#subdirectory=hi-ml-multimodal"
        ),
    ],  # Các thư viện phụ thuộc (nếu có)
    include_package_data=True,
    package_data={
        "kltn_utils": [
            "cbm/data/isic2018/*.json",
            "cbm/data/bccd/*.json",
            "cbm/data/busi/*.json",
            "cbm/data/dtr/*.json",
            "cbm/data/idrid/*.json",
            "cbm/data/lcc/*.json",
            "cbm/data/nct/*.json",
        ],
    },
)
