import numpy as np
import pandas as pd
from development_phase import evaluate
from torchgeo.datasets import QuakeSet


if __name__ == '__main__':
    # Load private test set
    dataset = QuakeSet(root="private_set", split="test")
    # Evaluate
    res_class, res_mag = np.ones(len(dataset)), np.zeros(len(dataset))
    predictions = []
    for metadata, sample in zip(dataset.data, dataset):
        cl, mg = evaluate(sample["image"])
        predictions += [
            {"key": metadata['key'], "magnitude": mg, "affected": int(cl == 1)}
        ]
    pd.DataFrame(predictions).to_csv("submission.csv", index=False)
