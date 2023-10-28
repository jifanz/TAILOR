import numpy as np


def random_sampling(trainer, model, embs, preds, labels, labeled, dataset, batch_size):
    labeled_set = set(list(labeled))
    all_set = set(list(range(len(dataset))))
    unlabeled = np.array(list(all_set - labeled_set))
    return np.random.choice(unlabeled, batch_size, replace=False)
