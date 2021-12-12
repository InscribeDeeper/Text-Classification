from transformers import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import ReLU


class clf_naive(nn.Module):
    # define all the layers used in model
    def __init__(self, emb_dim, seq_len, hidden_units, num_classes, dropout_rate=0):

        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        # self.cnn_bow_extractor = torch.nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(1, 3),
        # )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.emb_dim // 2, self.emb_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.emb_dim // 4, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_units, self.num_classes),
        )

    def forward(self, x):
        # depends on input of x, we can extracted from pre trained BERT first to saved memory
        # x = torch.mean(x, axis=1)  # (batchsize, seq_len, emb_dim) -> avg embedding to represent sentence
        # x = x[:, 0,:]  # (batchsize, emb_dim) -> first [CLS] embedding to represent sentence
        logit = self.fc(x)  # (N, num_classes)
        return logit


# Create the BertClassfier class
class clf_finetuneBERT(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, emb_dim=768, freeze_bert=False, num_classes=20, hidden_units=50):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(clf_finetuneBERT, self).__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_units = hidden_units

        self.classifier = nn.Sequential(
            nn.Linear(self.emb_dim, self.hidden_units),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(self.hidden_units, num_classes))

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # embed_type = 3 
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]  # [hidden state layer output][batch N, [CLS position], [embedding 768]]

        # embed_type = 2
        # outputs.pooler_output
        
        # embed_type = 1
        # average 
        # average_hidden_state_four_layers = torch.stack(hidden_states[2][-4:], dim=0).permute(1, 2, 0, 3).mean(axis=2)


        logits = self.classifier(last_hidden_state_cls)
        return logits


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

    def forward(self, x):
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
        return logit


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
