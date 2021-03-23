import os

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('Created {}'.format(dir_path))
        return

def acc(y, y_hat):
    return accuracy_score(y, y_hat)


def f1_macro(y, y_hat):
    return f1_score(y, y_hat, average='macro')


def precision_macro(y, y_hat):
    return precision_score(y, y_hat, average='macro')


def recall_macro(y, y_hat):
    return recall_score(y, y_hat, average='macro')

def number_h(num: object) -> object:
    for unit in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z']:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, 'Yi')

class EarlyStopping():
    def __init__(self, mode, patience):
        """
        :param mode: min or max
        :param patience: nof epochs to wait before stopping
        """
        self.mode = mode
        self.patience = patience
        self.current_patience = patience
        if self.mode == "max":
            self.best_metric = 0.0
        else:
            self.best_metric = 10000.0

    def stop(self, current_metric):
        if self.mode == "max":
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.current_patience = self.patience
            else:
                self.current_patience -= 1

            print("patience left:{}, best({})".format(self.current_patience, self.best_metric))
            print()

            if self.current_patience == 0:
                return True
            else:
                return False
        else:
            # mode = "min"
            if current_metric < self.best_metric:
                self.best_metric = current_metric
                self.current_patience = self.patience
            else:
                self.current_patience -= 1

            print("patience left:{}, best({})".format(self.current_patience, self.best_metric))
            print()

            if self.current_patience == 0:
                return True
            else:
                return False