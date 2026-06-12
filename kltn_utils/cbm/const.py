from importlib.resources import files

from . import utils

FOLDER_PATH = "kltn_utils.cbm.data"

CLASS_AND_CONCEPT = {
    "isic2018": utils.load_json(
        files(f"{FOLDER_PATH}.isic2018") / "class_concept.json"
    ),
    "lcc": utils.load_json(files(f"{FOLDER_PATH}.lcc") / "class_concept.json"),
    "nct": utils.load_json(files(f"{FOLDER_PATH}.nct") / "class_concept.json"),
    "idrid": utils.load_json(files(f"{FOLDER_PATH}.idrid") / "class_concept.json"),
    "busi": utils.load_json(files(f"{FOLDER_PATH}.busi") / "class_concept.json"),
}
