import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_stratified(labels, train_size, val_size):
    all_idx = np.arange(len(labels))

    test_size = 1.0 - train_size - val_size

    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=(val_size + test_size),
        stratify=labels,
        random_state=42,
        shuffle=True,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size / (test_size + val_size),
        stratify=labels[temp_idx],
        random_state=42,
        shuffle=True,
    )

    return train_idx, val_idx, test_idx


def train_test_stratified(labels, train_size):
    all_idx = np.arange(len(labels))

    test_size = 1.0 - train_size

    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        stratify=labels,
        random_state=42,
        shuffle=True,
    )

    return train_idx, test_idx
