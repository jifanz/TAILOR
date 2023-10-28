import numpy as np


class MultiClassMetric:
    def __init__(self):
        self.last_acc = None

    @staticmethod
    def __accuracy(preds, targets):
        n_class = preds.shape[1]
        accs = []
        preds_label = np.argmax(preds, axis=-1)
        targets_label = np.argmax(targets, axis=-1)
        correct = (preds_label == targets_label).astype(float)
        count = np.sum(targets, axis=0)
        for i in range(n_class):
            target = targets[:, i]
            acc = np.sum(target * correct) / max(count[i], 1)
            accs.append(acc)
        accs = np.array(accs)
        return np.mean(accs), accs

    def compute(self, epoch, preds, labels, losses, num_labeled=None, labeled=None):
        train_acc, train_accs = self.__accuracy(preds, labels)
        train_loss = np.mean(losses)
        if labeled is not None:
            num_pos = np.sum(labels[labeled], axis=0)
        else:
            num_pos = np.zeros(labels.shape[1])
        self.num_pos = num_pos
        self.dict = {"Epoch": epoch,
                     "Num Labeled": len(preds) if num_labeled is None else num_labeled,
                     "Training Accuracy": train_acc,
                     "Training Loss": train_loss,
                     "Total Number of Positive": np.sum(num_pos),
                     "Min Class Number of Positive": np.min(num_pos),
                     "Max Class Number of Positive": np.max(num_pos),
                     "Min Class Train Accuracy": np.min(train_accs),
                     "Max Class Train Accuracy": np.max(train_accs),
                     "STD Train Accuracy": np.std(train_accs),
                     }
        return self.dict


class MultiLabelMetric:
    def __init__(self):
        self.last_acc = None

    @staticmethod
    def __average_precision(pred, target):
        indices = np.argsort(-pred)
        pos_count = 0.
        precision_at_i = 0.
        for count, i in enumerate(indices):
            label = target[i]
            if label > .5:
                pos_count += 1
                precision_at_i += pos_count / (count + 1)
        return precision_at_i / max(pos_count, 1)

    def mean_average_precision(self, preds, targets, ret_ap=False):
        n_class = preds.shape[1]
        average_precisions = []
        for i in range(n_class):
            average_precisions.append(self.__average_precision(preds[:, i], targets[:, i]))
        average_precisions = np.array(average_precisions)
        if ret_ap:
            return np.mean(average_precisions), average_precisions
        else:
            return np.mean(average_precisions)

    @staticmethod
    def __evaluate(preds, targets):
        n, n_class = preds.shape
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        for k in range(n_class):
            pred = preds[:, k]
            target = targets[:, k]
            target[target < .5] = 0
            Ng[k] = np.sum(target > .5)
            Np[k] = np.sum(pred >= 0.5)
            Nc[k] = np.sum(target * (pred >= 0.5))
        Np[Np == 0] = 1
        OP = np.sum(Nc) / max(np.sum(Np), 1e-5)
        OR = np.sum(Nc) / max(np.sum(Ng), 1e-5)
        OF1 = (2 * OP * OR) / max((OP + OR), 1e-5)

        CP = np.sum(Nc / np.clip(Np, a_min=1e-5, a_max=None)) / n_class
        CR = np.sum(Nc / np.clip(Ng, a_min=1e-5, a_max=None)) / n_class
        CF1 = (2 * CP * CR) / max((CP + CR), 1e-5)
        return OP, OR, OF1, CP, CR, CF1

    @staticmethod
    def __accuracy(preds, targets):
        n_class = preds.shape[1]
        accs = []
        for i in range(n_class):
            pred = (preds[:, i] > .5)
            label = (targets[:, i] > .5)
            acc = np.mean((pred == label).astype(float))
            accs.append(acc)
        accs = np.array(accs)
        return np.mean(accs)

    def compute(self, epoch, preds, labels, losses, num_labeled=None, labeled=None):
        train_OP, train_OR, train_OF1, train_CP, train_CR, train_CF1 = self.__evaluate(preds, labels)
        train_acc = self.__accuracy(preds, labels)
        train_loss = np.mean(losses)
        train_map = self.mean_average_precision(preds, labels)
        if labeled is not None:
            num_pos = np.sum(labels[labeled], axis=0)
        else:
            num_pos = np.zeros(labels.shape[1])
        self.num_pos = num_pos
        self.dict = {"Epoch": epoch,
                     "Num Labeled": len(preds) if num_labeled is None else num_labeled,
                     "Training Overall Precision": train_OP,
                     "Training Overall Recall": train_OR,
                     "Training Overall F1": train_OF1,
                     "Training Class Average Precision": train_CP,
                     "Training Class Average Recall": train_CR,
                     "Training Class Average F1": train_CF1,
                     "Training Accuracy": train_acc,
                     "Training Loss": train_loss,
                     "Training mAP": train_map,
                     "Total Number of Positive": np.sum(num_pos),
                     "Min Class Number of Positive": np.min(num_pos),
                     "Max Class Number of Positive": np.max(num_pos),
                     }
        return self.dict
