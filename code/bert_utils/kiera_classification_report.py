import sys
import argparse
import os
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def parse_args():
    parser = argparse.ArgumentParser(description="report for different MAML")
    parser.add_argument("--MAML_model_name", type=str, default='meta_model_k3_qa', help="The file that MAML output the prediction with rid")
    parser.add_argument("--use_colab", type=bool, default=False, help="use colab or not")
    parser.add_argument("--device", type=str, default=None, help="use colab or not")
    parser.add_argument("--test_file", type=str, default='surprise_dt_test_v5_all_kiera.xlsx', help="use colab or not")
    #     args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args


args = parse_args()
MAML_model_name = args.MAML_model_name  # conf_sample_file = 'meta_model_v7.xlsx'

args.device = 'cpu'
conf_sample_file = MAML_model_name
test_file = args.test_file
## INIT

if args.use_colab:
    sys_path = '/content/drive/MyDrive/'
    from google.colab import drive
    drive.mount('/content/drive')
    # !pip install cloud-tpu-client==0.10 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
    #     !pip install torchinfo
    #     !pip install transformers
    #     !pip install learn2learn
    #     !pip install datasets
    #     !pip install pytorch
    # Import the os module
    # !pip install tqdm
    # os.system("pip install -U yellowbrick")
    # os.system("pip install linearmodels")

else:
    sys_path = 'E:/MyGoogleDrive/'

sys.path.append(sys_path + 'Conf_Call/code/5_emotion_extraction')
print(sys_path + 'Conf_Call/code/5_emotion_extraction')

import pandas as pd
import numpy as np
import pickle
from glob import glob
from tqdm import tqdm
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
# % matplotlib inline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from pprint import pprint
from random import sample

import torch
from torchinfo import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertModel, AdamW, BertConfig, get_linear_schedule_with_warmup
# , AutoModel, AutoTokenizer
import learn2learn as l2l

from wei_utils.training_utils import extract_contextual_embedding, train_multi_label_model, model_eval
from wei_utils.data_utils import re_softmax, label_encoding
from wei_utils import glovar
from wei_utils.data_loader import data_loading, BERT_data_loader, preprocessing_for_emo_mix_balance
from wei_utils.maml_training_utils import MAML_train, history_vis
from wei_utils.maml_dataloader import MAML_Data_loader, MAML_train_val_split

from emotion_classifier.model import clf, lstm_cnn_o2
from MAML.model import lstm_cnn_o1

## model setting
hidden_units = 200
embed_dim = 768
num_filters, kernel_sizes = 30, [1, 3, 5]
max_len = 100
learning_rate_alpha = 0.001
label_size = 2

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True, output_hidden_states=True)

# global device
if args.device is None:
    device = glovar.device_type
else:
    device = args.device

bert_model = bert_model.to(device)

print(next(bert_model.parameters()).device)  # 输出：cpu
# summary(bert_model)

model_path = os.path.join('.', 'models', MAML_model_name)
model = lstm_cnn_o1(embed_dim, max_len, hidden_units, num_filters, kernel_sizes, label_size)
# cnn_lstm = Base_LSTM_CNN(emb_dim=768, seq_len=200, lstm_units=200, num_filters=30, kernel_sizes=[1, 3, 5], num_classes=2)
# learning_rate_alpha = 0.01 #@param {type:"number"} # 0.001 ~

maml = l2l.algorithms.MAML(model, lr=learning_rate_alpha, first_order=False)
maml.module.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
maml = maml.to(device)
# maml = maml.half()
maml.eval()



# Load the dataset into a pandas dataframe.
if 'xlsx' in test_file:
    df = pd.read_excel(os.path.join('data', 'test', test_file))
elif 'txt' in test_file:
    df = pd.read_csv(os.path.join('data', 'test', test_file), sep='|')

df = df.dropna()
sentences = df.sentence.values
labels = df.label.values


# get embedding
_, _, x = extract_contextual_embedding(sentences, tokenizer, bert_model, max_len=max_len, device=device)
x = torch.tensor(x, dtype=torch.float32).to(device)
with torch.no_grad():
    preds = maml(x).view(-1).detach().cpu().numpy()

print("#" * 30)
print("Classfication report from model: ", MAML_model_name)
print(classification_report(labels, preds >= 0.5))
print("#" * 30)


with open(os.path.join('./', 'log__', 'report', conf_sample_file + '_classification_report_on_kiera.txt'), 'w') as f:
    print(classification_report(labels, preds >= 0.5), file=f)


fpr, tpr, thresholds = roc_curve(labels, preds >= 0.5)

ax = plt.figure(figsize=(20, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='MAML-qa')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No skill')
plt.title('AUC of ' + conf_sample_file)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend()
plt.show()
ax.savefig(os.path.join('./', 'log__', 'report', conf_sample_file + '_roc.jpeg'), dpi=400)
