from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image

from . import kltn_utils


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        transform,
        class_names,
    ):
        self.dataset_dir = dataset_dir
        self.transforms = transform

        self.file_paths, self.labels = kltn_utils.load_img_classify_data(
            dataset_dir, class_names
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])

        img = read_image(
            file_path,
            mode=ImageReadMode.RGB,
        )

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class ImageConceptDataset(Dataset):
    def __init__(self, dataset_dir, transform, class_names, concept2class):
        self.dataset_dir = dataset_dir
        self.transforms = transform

        self.file_paths, self.labels = kltn_utils.load_img_classify_data(
            dataset_dir, class_names
        )

        self.class2concept = kltn_utils.build_class_concept_matrix(
            concept2class, len(class_names)
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = int(self.labels[idx])
        concept = self.class2concept[label]

        img = read_image(
            file_path,
            mode=ImageReadMode.RGB,
        )

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label, concept
