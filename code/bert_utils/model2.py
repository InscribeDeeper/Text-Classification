import torch.nn as nn
import torch
import torch.nn.functional as F


class clf(nn.Module):

    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, lstm_units, num_classes, dropout_rate=0.2):

        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.num_classes = num_classes

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.lstm = nn.LSTM(
            emb_dim,
            lstm_units,
            num_layers=2,
            bidirectional=True,
            dropout=0.1,  # for multiple layers
            batch_first=True)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(2 * lstm_units, self.num_classes),
        )

#         self._initialization()

    def forward(self, x):
        # <------ packed_output 是 ([32, 200, 160]) 需要用 CNN 1D 去提取? ------>

        # encoder x,  eval() 和 train() 不会影响

        packed_output, (h_T, c_T) = self.lstm(x)  # (N, seq_len, 2*lstm_units)
        # h_T = [N, num layers * num directions, hid dim] => 最后一个 timestamp 输出的 vector
        # c_T = [N, num layers * num directions, hid dim] => 最后一个 timestamp 输出的 vector
        # packed_output = N, seq_len, num_directions * hidden_size
        # print(packed_output[:,-1,:].size())
        hidden = torch.cat((h_T[-2, :, :], h_T[-1, :, :]), dim=1)  # 取最后两层, 测试是否 work # 80% acc
        # hidden = torch.cat((c_T[-2,:,:], c_T[-1,:,:]), dim = 1) # 取最后两层, 测试是否 work # 81% acc
        # hidden = packed_output[:,-1,:] # [4, 16, 160]
        # hidden = packed_output[:,-1,:] # [4, 16, 160]
        # print(hidden.size())

        logit = self.fc(hidden)  # (N, num_classes)
        # prop = torch.sigmoid(logit)  # Sigmoid for multilabel prediction + BCEWithLogitsLoss
        # print(prop.shape)

        #         print(h_T.shape)
        #         print(hidden.shape)
        #         print(packed_output.shape)
        return logit


# class SentimentNet(nn.Module):
#     def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
#         super(SentimentNet, self).__init__()
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.hidden_dim = hidden_dim

#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim,
#                             n_layers, dropout=drop_prob, batch_first=True)
#         self.dropout = nn.Dropout(drop_prob)
#         self.fc = nn.Linear(hidden_dim, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x, hidden):
#         batch_size = x.size(0)
#         x = x.long()
#         embeds = self.embedding(x)
#         lstm_out, hidden = self.lstm(embeds, hidden)
#         lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

#         out = self.dropout(lstm_out)
#         out = self.fc(out)
#         out = self.sigmoid(out)

#         out = out.view(batch_size, -1)
#         out = out[:, -1]
#         return out, hidden

#     def init_hidden(self, batch_size):
#         weight = next(self.parameters()).data
#         hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
#                   weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
#         return hidden

# hidden_units = 80
# embed_dim = 768

# patience = 10
# model_path='./saved_models/clf_surprise.pt'
# epochs = 10

# max_len = 200
# batch_size = 16
# label_size = 2
# model = clf(embed_dim, max_len, hidden_units, label_size)
# # model.to(device)
# from torchinfo import summary
# print(summary(model,(batch_size, max_len, embed_dim)))


class lstm_cnn_o2(nn.Module):
    '''The output dimension is two, use the BCE loss with logit'''

    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, lstm_units, num_filters, kernel_sizes, num_classes, dropout_rate=0.5):

        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes  # [1,2,3] -> filter size
        self.num_classes = num_classes

        # self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        self.lstm = nn.LSTM(emb_dim, lstm_units, num_layers=1, bidirectional=True, batch_first=True)

        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (f, 2 * self.lstm_units)) for f in self.kernel_sizes])
        self.fc = nn.Linear(len(kernel_sizes) * self.num_filters, self.num_classes)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):

        x, _ = self.lstm(x)  # (N, seq_len, 2*lstm_units)

        x = x.unsqueeze(1)

        # print(x.size())

        x = [F.relu(conv(x).squeeze(-1)) for conv in self.convs]  # output of three conv

        # print(x[0].size())

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # continue with 3 maxpooling

        x = torch.cat(x, 1)  # N, len(filter_sizes)* num_filters
        # print(x.size())

        x = self.dropout(x)  # N, len(filter_sizes)* num_filters

        logit = self.fc(x)  # (N, num_classes)

        return logit


class RNN(nn.Module):
    lr = 0.0005

    def __init__(self, input_size, hidden_size, embeding_size, n_categories, n_layers, output_size, p):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.embeding = nn.Embedding(input_size + n_categories, embeding_size)
        self.lstm = nn.LSTM(embeding_size + n_categories, hidden_size, n_layers, dropout=p)
        self.out_fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p)

    def forward(self, batch_of_category, batch_of_letter, hidden, cell):
        ## letter level operations

        embeding = self.dropout(self.embeding(batch_of_letter))
        category_plus_letter = torch.cat((batch_of_category, embeding), 1)

        # sequence_length = 1
        category_plus_letter = category_plus_letter.unsqueeze(1)

        out, (hidden, cell) = self.lstm(category_plus_letter, (hidden, cell))
        out = self.out_fc(out)
        out = out.squeeze(1)

        return out, (hidden, cell)

    def configure_optimizers(self):
        # optimizer = Adam(self.parameters(), self.lr)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # return [optimizer], [scheduler]
        pass

    def training_step(self, batch, batch_idx):
        item_dict = batch
        loss = 0
        batch_of_category = item_dict["category_tensors"]

        # we loop over letters, single batch at the time
        criterion = None
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size).cuda()
        cell = torch.zeros(self.n_layers, 1, self.hidden_size).cuda()

        for t in range(item_dict["input_tensors"].size(1)):
            batch_of_letter = item_dict["input_tensors"][:, t]

            output, (hidden, cell) = self(batch_of_category, batch_of_letter, hidden, cell)

            loss += criterion(output, item_dict["target_tensors"][:, t])

        loss = loss / (t + 1)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size)

        return hidden, cell

