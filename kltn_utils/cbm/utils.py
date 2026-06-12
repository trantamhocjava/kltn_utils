import json


def load_json(package_file):
    with package_file.open("r", encoding="utf-8") as f:
        return json.load(f)
