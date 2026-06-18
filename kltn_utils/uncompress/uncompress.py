import gzip
import shutil
import tarfile
import zipfile


def uncompress_gzip(src_file_path, dst_file_path):
    """
    file .gz
    """
    try:
        with gzip.GzipFile(src_file_path, "rb") as f_in:
            with open(dst_file_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        print(f"Uncompress {src_file_path} to {dst_file_path} OK")
    except Exception as e:
        print(f"Uncompress {src_file_path} to {dst_file_path} error: {str(e)}")


def extract_tar(src_file_path, dst_dir_path):
    """
    file .tar, .tar.gz, .tgz, .tar.bz2
    """
    try:
        with tarfile.open(src_file_path, "r:*") as tar:
            tar.extractall(path=dst_dir_path)

        print(f"Extract {src_file_path} to {dst_dir_path} OK")
    except Exception as e:
        print(f"Extract {src_file_path} to {dst_dir_path} error: {str(e)}")


def extract_zip(zip_path, extract_dir):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    print(f"Extracted to: {extract_dir}")
