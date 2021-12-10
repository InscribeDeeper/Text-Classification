import torch.nn as nn
import torch
import torch.nn.functional as F


class lstm_cnn_o1(nn.Module):
    '''The output dimension is one, use the torch.nn.BCELoss(reduction='mean')'''

    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, lstm_units, num_filters, kernel_sizes, num_classes, dropout_rate=0.5):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.lstm_units = lstm_units
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        self.lstm = nn.LSTM(emb_dim, lstm_units, num_layers=1, bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filters, (f, 2 * self.lstm_units)) for f in self.kernel_sizes])
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.fc = nn.Sequential(nn.Linear(len(kernel_sizes) * self.num_filters, 1), nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(len(kernel_sizes) * self.num_filters, self.num_classes))

    def forward(self, x):
        x, _ = self.lstm(x)  # (N, seq_len, 2*lstm_units)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x).squeeze(-1)) for conv in self.convs]  # output of three conv
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # continue with 3 maxpooling
        x = torch.cat(x, 1)  # N, len(filter_sizes)* num_filters
        x = self.dropout(x)  # N, len(filter_sizes)* num_filters
        logit = self.fc(x)  # (N, 1)

        return logit
