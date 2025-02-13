from typing import List

import numpy as np
from sklearn.model_selection import KFold


def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    # import pdb;pdb.set_trace()
    ## splits
    # [
    #     {'train': ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'], 'val': ['A', 'J']},
    #     {'train': ['A', 'C', 'D', 'E', 'F', 'H', 'I', 'J'], 'val': ['B', 'G']},
    #     {'train': ['A', 'B', 'D', 'E', 'G', 'H', 'I', 'J'], 'val': ['C', 'F']},
    #     {'train': ['A', 'B', 'C', 'E', 'F', 'G', 'I', 'J'], 'val': ['D', 'H']},
    #     {'train': ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'J'], 'val': ['E', 'I']}
    # ]

    return splits
