import numpy as np
from sklearn.utils import shuffle
import pandas as pd


class MAML_Data_loader():

    # data is a list with all feature arrays
    # X_train_pos, X_train_neg, X_val_pos,X_val_neg only
    # contain indexes for train and validation
    def __init__(self, X_train_pos, X_train_neg, X_val_pos, X_val_neg, data, batch_size, k_shot=1, train_mode=True):

        self.data = data

        self.batch_size = batch_size
        # self.n_way = n_way  # 5 or 20, how many classes the model has to select from
        self.k_shot = k_shot  # 1 or 5, how many times the model sees the example

        self.num_classes = 2

        self.train_pos = X_train_pos
        self.train_neg = X_train_neg

        # position of last batch
        self.train_pos_index = 0
        self.train_neg_index = 0

        if not train_mode:

            self.val_pos = X_val_pos
            self.val_neg = X_val_neg

            self.val_pos_index = 0
            self.val_neg_index = 0

            # merge train & val for prediction use
            self.all_pos = np.concatenate([self.train_pos, self.val_pos])
            self.all_neg = np.concatenate([self.train_neg, self.val_neg])

            self.pos_index = 0
            self.neg_index = 0

        self.iters = 100

    def next_batch(self, dtype=np.int32):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []

        x_set = []
        y_set = []

        for _ in range(self.batch_size):

            x_set = []
            y_set = []

            target_class = np.random.randint(self.num_classes)
            # print(target_class)

            # negative class
            for i in range(self.k_shot + 1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_neg_index == len(self.train_neg):
                    self.train_neg = np.random.permutation(self.train_neg)
                    self.train_neg_index = 0
                    # print("neg seq", self.train_neg_seq)

                if i == self.k_shot:  # the last one is test sample
                    if target_class == 0:  # positive class
                        x_hat_batch.append(self.train_neg[self.train_neg_index])
                        y_hat_batch.append(0)
                        self.train_neg_index += 1
                else:
                    x_set.append(self.train_neg[self.train_neg_index])
                    y_set.append(0)
                    self.train_neg_index += 1

            # positive class
            for i in range(self.k_shot + 1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_pos_index == len(self.train_pos):

                    self.train_pos = np.random.permutation(self.train_pos)
                    self.train_pos_index = 0
                    # print("pos seq", self.train_pos_seq)

                if i == self.k_shot:  # the last one is test sample

                    if target_class == 1:  # positive class
                        x_hat_batch.append(self.train_pos[self.train_pos_index])

                        y_hat_batch.append(1)
                        self.train_pos_index += 1

                else:
                    x_set.append(self.train_pos[self.train_pos_index])

                    y_set.append(1)
                    self.train_pos_index += 1

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        # get feature arrays for the batch

        # print(x_set_batch)
        # print(x_hat_batch)

        feature_set_batch = []
        feature_hat_batch = []

        for feature in self.data:

            f_set = np.array([np.array(feature[b]) for b in x_set_batch])
            f_hat = np.array(feature[x_hat_batch])
            # print(f_set.shape)
            # print(f_hat.shape)

            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)

        return feature_set_batch, np.asarray(y_set_batch).astype(dtype), feature_hat_batch, np.asarray(y_hat_batch).astype(dtype), np.zeros(self.batch_size)  # all 0s for aux output

    def next_eval_batch(self, dtype=np.int32):
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []
        y_hat_batch = []

        for _ in range(self.batch_size):

            x_set = []
            y_set = []

            target_class = np.random.randint(self.num_classes)
            # print(target_class)

            if self.val_pos_index == len(self.val_pos):
                self.val_pos = np.random.permutation(self.val_pos)
                self.val_pos_index = 0
                # print("pos val seq", self.val_pos_seq)

            if self.val_neg_index == len(self.val_neg):
                self.val_neg = np.random.permutation(self.val_neg)
                self.val_neg_index = 0
                # print("net val seq", self.val_neg_seq)

            # negative class
            for i in range(self.k_shot + 1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_neg_index == len(self.train_neg):
                    self.train_neg = np.random.permutation(self.train_neg)
                    self.train_neg_index = 0
                    # print("neg seq", self.train_neg_seq)

                if i == self.k_shot:  # the last one is test sample

                    if target_class == 0:  # negative class
                        x_hat_batch.append(self.val_neg[self.val_neg_index])
                        y_hat_batch.append(0)
                        self.val_neg_index += 1
                else:

                    x_set.append(self.train_neg[self.train_neg_index])
                    y_set.append(0)
                    self.train_neg_index += 1

            # positive class
            for i in range(self.k_shot + 1):

                # shuffle pos or neg if a sequence has been full used
                if self.train_pos_index == len(self.train_pos):
                    self.train_pos = np.random.permutation(self.train_pos)
                    self.train_pos_index = 0
                    # print("pos seq", self.train_pos_seq)

                if i == self.k_shot:  # the last one is test sample

                    if target_class == 1:  # positive class
                        x_hat_batch.append(self.val_pos[self.val_pos_index])

                        y_hat_batch.append(1)
                        self.val_pos_index += 1

                else:
                    x_set.append(self.train_pos[self.train_pos_index])

                    y_set.append(1)
                    self.train_pos_index += 1

            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        # print(x_set_batch)
        # print(x_hat_batch)

        feature_set_batch = []
        feature_hat_batch = []

        # loop through all features
        for feature in self.data:

            f_set = np.array([np.array(feature[b]) for b in x_set_batch])
            f_hat = np.array(feature[x_hat_batch])
            # print(f_set.shape)
            # print(f_hat.shape)

            feature_set_batch.append(f_set)
            feature_hat_batch.append(f_hat)

        return feature_set_batch, np.asarray(y_set_batch).astype(dtype), feature_hat_batch, np.asarray(y_hat_batch).astype(dtype), np.zeros(self.batch_size)  # all 0s for aux output

    # generate support set for each sample in prediction
    # use all samples as support
    def get_pred_set(self, pred, dtype=np.int32):  # new sentence
        x_set_batch = []
        y_set_batch = []
        x_hat_batch = []

        for _ in range(self.batch_size):  # batch_size = 32

            x_set = []
            y_set = []

            # target_class = np.random.randint(self.num_classes)  # target_class = 0/1
            # print(target_class)

            if self.pos_index == len(self.all_pos):  # initiate the index
                self.all_pos = np.random.permutation(self.all_pos)  # shuffle
                self.pos_index = 0

            if self.neg_index == len(self.all_neg):  # initiate the index
                self.all_neg = np.random.permutation(self.all_neg)
                self.neg_index = 0

            # negative class
            for i in range(self.k_shot):

                # shuffle pos or neg if a sequence has been full used
                if self.neg_index == len(self.all_neg):
                    self.all_neg = np.random.permutation(self.all_neg)
                    self.neg_index = 0
                    # print("neg seq", self.train_neg_seq)

                x_set.append(self.all_neg[self.neg_index])

                y_set.append(0)
                self.neg_index += 1

            # positive class
            for i in range(self.k_shot):

                # shuffle pos or neg if a sequence has been full used
                if self.pos_index == len(self.all_pos):
                    self.all_pos = np.random.permutation(self.all_pos)
                    self.pos_index = 0
                    # print("pos seq", self.train_pos_seq)

                x_set.append(self.all_pos[self.pos_index])

                y_set.append(1)
                self.pos_index += 1

            # Prediction sample
            x_set_batch.append(x_set)
            y_set_batch.append(y_set)

        x_hat_batch.append(pred)

        # repeat each element in pred for batch_size times
        feature_hat_batch = [np.repeat(e, self.batch_size, axis=0) for e in pred]
        feature_set_batch = []

        # loop through all features
        for idx, feature in enumerate(self.data):

            f_set = np.array([np.array(feature[b]) for b in x_set_batch])
            # print(f_set.shape)
            feature_set_batch.append(f_set)

        return feature_set_batch, np.asarray(y_set_batch).astype(dtype), feature_hat_batch  # x, y, x_query

    # def get_pred_set_gen(self, pred):
    #    while True:
    #        x_set, y_set, x_hat, y_hat = train_loader.next_batch()
    #        yield([x_set, x_hat], 1-y_hat)

    def next_eval_batch_gen(self):
        while True:
            x_set, y_set, x_hat, y_hat, aux_y = self.next_eval_batch()
            yield (x_set + x_hat, [y_set, y_hat])

    def next_batch_gen(self):
        while True:

            x_set, y_set, x_hat, y_hat, aux_y = self.next_batch()

            yield (x_set + x_hat, [y_set, y_hat])

    def next_eval_batch_gen_context(self):
        while True:
            x_set, y_set, x_hat, y_hat = self.next_eval_batch()
            yield (x_set + x_hat, [y_set, y_hat])

    def next_batch_gen_context(self):
        while True:

            x_set, y_set, x_hat, y_hat = self.next_batch()

            yield (x_set + x_hat, y_hat)


def MAML_train_val_split(df, sources):

    # splite train (2/3) and validation (1/3)
    frac = 2 / 3
    # random seed
    # sentence indexes for train
    Train_pos_set = {}
    Train_neg_set = {}

    # sentence indexes for validation validation
    Val_pos_set = {}
    Val_neg_set = {}

    for name in sources:
        print("\n", name)
        # X_train = []
        # X_val = []

        # positive case
        x = df[(df["source"] == name) & (df["label"] == 1)].index.values
        # x = shuffle(x, )
        x = shuffle(x, random_state=0)
        # print(x)

        # split train and validation data in positive case
        N = int(len(x) * frac)
        pos_train = x[0:N]  # Only sentence ids are stored
        pos_val = x[N:]

        # print("pos_train", len(pos_train), pos_train)
        # print("pos_val", len(pos_val), pos_val)

        # Negative case
        neg = df[(df["source"] == name) & (df["label"] == 0)].index.values
        neg = shuffle(neg, random_state=0)

        # split train and validation data in negative case
        N = int(len(neg) * frac)
        neg_train = neg[0:N]
        neg_val = neg[N:]
        # print("neg_train", len(neg_train), neg_train)
        # print("neg_val", len(neg_val), neg_val )

        # store the indexes into a diction
        Train_pos_set[name] = pos_train
        Train_neg_set[name] = neg_train

        Val_pos_set[name] = pos_val
        Val_neg_set[name] = neg_val

        if name == 'qa':
            pd.concat([df.loc[neg_train], df.loc[pos_train]], axis=0).to_csv('./data/test/maml_train_qa.txt', sep='|')
            pd.concat([df.loc[neg_val], df.loc[pos_val]], axis=0).to_csv('./data/test/maml_valid_qa.txt', sep='|')

    return Train_pos_set, Train_neg_set, Val_pos_set, Val_neg_set
