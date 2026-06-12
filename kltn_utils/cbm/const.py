from .. import kltn_utils

FOLDER_PATH = "kltn_utils/cbm"

CLASS_AND_CONCEPT = {
    "isic2018": kltn_utils.read_json_to_dict(
        f"{FOLDER_PATH}/data/isic2018/class_concept.json"
    ),
    "lcc": kltn_utils.read_json_to_dict(f"{FOLDER_PATH}/data/lcc/class_concept.json"),
    "nct": kltn_utils.read_json_to_dict(f"{FOLDER_PATH}/data/nct/class_concept.json"),
    "idrid": kltn_utils.read_json_to_dict(
        f"{FOLDER_PATH}/data/idrid/class_concept.json"
    ),
    "busi": kltn_utils.read_json_to_dict(f"{FOLDER_PATH}/data/busi/class_concept.json"),
}
