import shutil


def compress2zip(folder_path, zip_path):
    """
    zip_path: only name without extension, ex: zipped_thing
    """
    shutil.make_archive(
        base_name=zip_path,  # tạo checkpoint.zip
        format="zip",
        root_dir=folder_path,
    )

    print(f"Created: {zip_path}.zip ok")
