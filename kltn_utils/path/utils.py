import os
import shutil
from pathlib import Path


def get_file_name(file_path):
    """
    file_name_ext: file name with ext, ex: hello.png
    file_name: file name without ext, ex: hello
    """
    _, file_name_ext = os.path.split(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    return file_name_ext, file_name


def get_extension(file_path: str) -> str:
    return Path(file_path).suffix


def copy_files(src_folder_path, des_folder_path, file_names):
    for file_name in file_names:
        shutil.copy(f"{src_folder_path}/{file_name}", f"{des_folder_path}/{file_name}")
