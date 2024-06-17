import numpy as np
import pandas as pd
from tqdm import tqdm
from hashlib import sha256
from torchgeo.datasets import QuakeSet
from Data.LoadData import read_partition


def evaluate(X_in):
    if X_in.ndim == 4:
        X_in = X_in[0, :, :, :]
    mask = (np.array(X_in[1, :, :] == X_in[3, :, :]) * 1.0).astype(np.float32)
    mask2 = (np.array(X_in[0, :, :] == X_in[2, :, :]) * 1.0).astype(np.float32)
    # Find NaN locations
    condition = (X_in[1, :, :] == 0) & (X_in[0, :, :] == 0)
    mask[condition] = False
    mask2[condition] = False
    mask = mask.flatten()
    mask2 = mask2.flatten()
    sum_mask = np.sum(mask)
    len_mask = len(mask)
    sum_mask2 = np.sum(mask2)
    len_mask2 = len(mask2)

    # Classification logic
    classified, magnitude = 1, 0
    if (sum_mask / len_mask * 100 + sum_mask2 / len_mask2 * 100) > 7:
        classified = 0
    else:
        magnitude = 4.6

    return classified, magnitude


if __name__ == '__main__':
    # Method 1: Evaluate test for submission
    # Load test set
    dataset = QuakeSet(root="data", split="test", download=True)
    # Evaluate
    res_class, res_mag = np.ones(len(dataset)), np.zeros(len(dataset))
    predictions = []
    for i, sample in tqdm(enumerate(dataset)):
        cl, mg = evaluate(sample["image"])
        res_class[i], res_mag[i] = cl, mg
        metadata = dataset.data[i]
        # Note: The key generation made in this way for public evaluation only.
        key = f"{metadata['key']}/{metadata['patch']}/{metadata['images'][1]}"
        key = sha256(key.encode()).hexdigest()
        predictions += [
            {"key": key, "magnitude": mg, "affected": int(cl == 1)}
        ]
    pd.DataFrame(predictions).to_csv("submission.csv", index=False)

    # Method 2: Read partition with ground-truth and obtain metrics
    # Load test set
    images_test, labels_test, magnitudes_test = read_partition(split='test')
    # Evaluate
    res_class, res_mag = np.ones(len(labels_test)), np.zeros(len(labels_test))
    for i in range(len(labels_test)):
        cl, mg = evaluate(images_test[i, :, :, :])
        res_class[i], res_mag[i] = cl, mg
    # Performance
    print('Accuracy = ', np.sum(labels_test == res_class) / len(labels_test))
    from sklearn.metrics import f1_score
    print('F1 score = ', f1_score(labels_test, res_class, average='macro'))
    print('MAE = ', np.sum(np.abs(magnitudes_test - res_mag)) / len(labels_test))
