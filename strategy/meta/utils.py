from strategy.meta.emal_sub import EMALSubProcedure
from strategy.meta.uncertainty_sub import UncertaintySubProcedure
from strategy.meta.mlp_sub import MLPSubProcedure
from strategy.meta.galaxy_sub import Node, GALAXYSubProcedure
from strategy.meta.weaksup_sub import WeakSupSubProcedure
from strategy.meta.confidence_sub import ConfidenceSubProcedure
from strategy.meta.min_dist_galaxy_sub import MinDistGALAXYSubProcedure
from strategy.meta.badge_sub import BADGESubProcedure
from strategy.meta.similar_sub import SMISubProcedure, get_balanced_set
import numpy as np


def get_grad_embeddings(embs, preds):
    max_inds = np.argmax(preds, axis=-1)
    n = embs.shape[0]
    embeddings = -1 * preds
    embeddings[np.arange(n), max_inds] += 1
    return embeddings


def get_subprocedure(sub_names, trainer, model, embs, preds, labels, labeled, dataset, batch_size):
    if ("badge" in sub_names) or ("similar" in sub_names):
        all_set = set(list(range(len(dataset))))
        labeled_set = set(list(labeled))
        unlabeled = np.array(list(all_set - labeled_set))
        unlabeled = unlabeled[np.random.permutation(len(unlabeled))]
        labeled_subset = labeled[np.random.permutation(len(labeled))]
        labeled_subset = get_balanced_set(labeled_subset, labels, trainer.n_class, 5)
        idxs = np.concatenate((unlabeled, labeled_subset), axis=0)
        grad_embeddings = get_grad_embeddings(embs[idxs], preds[idxs])
        print(grad_embeddings.shape)
    sub_ps = []
    for name in sub_names:
        if name == "emal":
            sub_ps.append(EMALSubProcedure(trainer, model, embs, preds, labels, labeled, dataset, batch_size))
        elif name == "weak":
            sub_ps.append(WeakSupSubProcedure(trainer, model, embs, preds, labels, labeled, dataset, batch_size))
        elif name == "confidence":
            sub_ps.append(ConfidenceSubProcedure(trainer, model, embs, preds, labels, labeled, dataset, batch_size))
        elif name == "badge":
            sub_ps.append(
                BADGESubProcedure(trainer, model, embs, preds, labels, labeled, dataset, batch_size,
                                  grad_embeddings.reshape((grad_embeddings.shape[0], -1))[:len(unlabeled)], unlabeled))
        elif name == "similar":
            args = {"smi_function": "fl2mi",
                    # Use a facility location function, which captures representation information
                    "metric": "cosine",  # Use cosine similarity when determining the likeness of two data points
                    "optimizer": "LazyGreedy"
                    # When doing submodular maximization, use the lazy greedy optimizer
                    }
            sub_ps.append(
                SMISubProcedure(trainer, model, embs, preds, labels, labeled, dataset, batch_size, args,
                                grad_embeddings, unlabeled, labeled_subset))
        elif name == "uncertain":
            n_class = preds.shape[1]
            for i in range(n_class):
                sub_ps.append(UncertaintySubProcedure(trainer, model, embs, preds[:, i], labels[:, i], labeled, dataset,
                                                      batch_size, i))
        elif name == "mlp":
            n_class = preds.shape[1]
            for i in range(n_class):
                sub_ps.append(
                    MLPSubProcedure(trainer, model, embs, preds[:, i], labels[:, i], labeled, dataset, batch_size, i))
        elif name == "galaxy":
            n_class = preds.shape[1]
            nodes = []
            print(preds.shape, labels.shape)
            if trainer.multi_label_flag:
                margins = preds
            else:
                most_confident = np.max(preds, axis=1).reshape((-1, 1))
                margins = preds - most_confident + 1e-8 * most_confident
            for idx, (margin, label) in enumerate(zip(margins, labels)):
                nodes.append(Node(idx, margin, label))
            for i in labeled:
                nodes[i].update()
            for i in range(n_class):
                sub_ps.append(
                    GALAXYSubProcedure(trainer, model, embs, margins[:, i], labels[:, i], labeled, dataset, batch_size,
                                       nodes, i))
        elif name == "min_dist_galaxy":
            nodes = []
            print(preds.shape, labels.shape)
            if trainer.multi_label_flag:
                margins = preds
            else:
                most_confident = np.max(preds, axis=1).reshape((-1, 1))
                margins = preds - most_confident + 1e-8 * most_confident
            for idx, (margin, label) in enumerate(zip(margins, labels)):
                nodes.append(Node(idx, margin, label))
            for i in labeled:
                nodes[i].update()
            sub_ps.append(
                MinDistGALAXYSubProcedure(trainer, model, embs, margins, labels, labeled, dataset, batch_size, nodes))

    assert len(sub_ps) > 0
    print("Number of Algorithms: %d" % len(sub_ps))
    return sub_ps
