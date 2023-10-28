from strategy.meta.sub_procedure import SamplingSubProcedure
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy import stats
import pdb


def distance(X1, X2, mu):
    Y1, Y2 = mu
    X1_vec, X1_norm = X1
    X2_vec, X2_norm = X2
    Y1_vec, Y1_norm = Y1
    Y2_vec, Y2_norm = Y2
    dist = np.sqrt(np.clip(
        X1_norm ** 2 * X2_norm ** 2 + Y1_norm ** 2 * Y2_norm ** 2 - 2 * (X1_vec @ Y1_vec) * (X2_vec @ Y2_vec), a_min=0,
        a_max=None))
    return dist


def init_centers(X1, X2, unlabeled, chosen, mu, D2):
    if len(chosen) == 0:
        ind = np.argmax([norm1 * norm2 for norm1, norm2 in zip(X1[1], X2[1])])
        mu = [((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind]))]
        chosen = [ind]
    chosen_set = set(chosen)
    if len(mu) == 1:
        D2 = distance(X1, X2, mu[0]).ravel().astype(float)
    else:
        newD = distance(X1, X2, mu[-1]).ravel().astype(float)
        D2 = np.minimum(D2, newD)
    D2[chosen] = 0
    D2_unlabel = D2[unlabeled]
    print(str(len(mu)) + '\t' + str(sum(D2_unlabel)), flush=True)
    D2_unlabel = D2_unlabel.ravel().astype(float)
    Ddist = (D2_unlabel ** 2) / sum(D2_unlabel ** 2)
    customDist = stats.rv_discrete(name='custm', values=(np.arange(len(unlabeled)), Ddist))
    ind = unlabeled[customDist.rvs(size=1)[0]]
    assert ind not in chosen_set, "%d, %s" % (ind, str(chosen_set))
    mu.append(((X1[0][ind], X1[1][ind]), (X2[0][ind], X2[1][ind])))
    chosen.append(ind)
    del D2_unlabel
    return chosen, mu, D2


class BADGESubProcedure(SamplingSubProcedure):
    def __init__(self, trainer, model, embs, preds, labels, labeled, dataset, batch_size, grad_embeddings, unlabeled):
        super(BADGESubProcedure, self).__init__(trainer, model, embs, preds, labels, labeled, dataset, batch_size)
        self.grad_embeddings = grad_embeddings
        self.preds = preds
        self.batch_size = batch_size
        self.chosen = []
        self.n = len(dataset)
        self.mu = None
        self.D2 = None
        assert grad_embeddings.shape[0] == len(unlabeled)
        self.unlabeled_set = set(unlabeled)
        self.idx2i = {}
        for i, idx in enumerate(unlabeled):
            self.idx2i[idx] = i
        self.unlabeld_idx = unlabeled
        self.grad_norms = np.linalg.norm(self.grad_embeddings, axis=-1)
        self.embs = self.embs[unlabeled]
        self.emb_norms = np.linalg.norm(self.embs, axis=-1)

    def sample(self, labeled_set):
        unlabeled_set = self.unlabeled_set - labeled_set
        i_lst = np.array([self.idx2i[idx] for idx in unlabeled_set])
        self.chosen, self.mu, self.D2 = init_centers((self.grad_embeddings, self.grad_norms),
                                                     (self.embs, self.emb_norms), i_lst, self.chosen, self.mu, self.D2)
        return self.unlabeld_idx[self.chosen[-1]]

    def __str__(self):
        return "BADGE"
