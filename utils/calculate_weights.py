import os
from tqdm import tqdm
import numpy as np

from mypath import Path


def calculate_weights_labels(dataset, dataloader, num_classes):
    # Create an instance from th data loader
    z = np.zeros((num_classes,))

    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')

    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l

    tqdm_batch.close()
    total_frequency = np.sum(z)

    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)

    ret = np.array(class_weights)
    class_weights_path = os.path.join(Path.db_root_dir(dataset), dataset + '_classes_weights.npy')
    np.save(class_weights_path, ret)

    return ret
