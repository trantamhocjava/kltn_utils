import os


def get_file_name(file_path):
    _, file_name_ext = os.path.split(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    return file_name_ext, file_name
