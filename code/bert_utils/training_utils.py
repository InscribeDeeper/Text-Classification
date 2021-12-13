### This file prepared is prepared in my previously projects.
import random
import datetime
import torch
import numpy as np
import sys
from torch.nn.functional import softmax
from torch.nn import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score  # average_precision_score
import time
import copy
import os
from tqdm import tqdm

try:
    import glovar
except ImportError:
    print("Append path: ", os.path.abspath(os.path.dirname(__file__)))
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    import glovar

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return


def extract_contextual_embedding(sentences, tokenizer, bert_model, finetune=False, max_len=100, device=glovar.device_type, low_RAM_inner_batch=False, embed_type=2):
    '''
    return
        token_embeddings, input_ids, output_embedding
        one embedding sample, all input input_ids, all encoded sentences embedding
    '''
    # device = glovar.device_type

    input_ids = []
    attention_masks = []
    output_embedding = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent, truncation=True, add_special_tokens=True, max_length=max_len, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    if finetune:
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        output_embedding = input_ids  # for further processing.
        return input_ids, output_embedding, attention_masks

    else:
        # attention_masks = torch.cat(attention_masks, dim=0)
        # bert_model.eval()

        if low_RAM_inner_batch:
            with torch.no_grad():
                # too large -> for loop to V-RAM handle
                for batch_input_ids, batch_attention_masks in tqdm(list(zip(input_ids, attention_masks))):
                    # for batch_input_ids, batch_attention_masks in (list(zip(input_ids, attention_masks))):
                    outputs = bert_model(batch_input_ids.to(device), batch_attention_masks.to(device))
                    ## in output
                    # 0.last_hidden_state
                    # 1.pooler_output
                    # 2.hidden_states
                    # 3.attentions

                    ## dimension
                    # {'attentions': 12 layer * torch.Size([1, 12, 100, 100]),
                    # 'hidden_states': (12 layer + 1 output layer) *  torch.Size([1, 100, 768]),
                    # 'last_hidden_state': torch.Size([100, 768]),
                    # 'pooler_output': torch.Size([768])}

                    if embed_type == 1:  # to get contextual embedding for each words
                        hidden_states = outputs[2]
                        token_embeddings = torch.stack(hidden_states[-4:], dim=0)  # get the last four layers
                        token_embeddings = token_embeddings.permute(1, 2, 0, 3)
                        token_embeddings = token_embeddings.mean(axis=2)
                        token_embeddings = (token_embeddings.cpu().numpy()) * (torch.tile(batch_attention_masks, (768, 1)).T)
                    elif embed_type == 2:
                        token_embeddings = outputs.pooler_output
                    elif embed_type == 3:
                        token_embeddings = outputs.last_hidden_state[0][0]  # [CLS] tokens
                    output_embedding.append(token_embeddings.cpu().numpy())

            input_ids = torch.cat(input_ids, dim=0)  # Convert the lists into tensors.

            if embed_type == 2:
                output_embedding = np.concatenate(output_embedding, axis=0)
            else:
                output_embedding = np.array(output_embedding)

        else:  # high RAM outer batch
            with torch.no_grad():
                # Convert the lists into tensors.
                input_ids = torch.cat(input_ids, dim=0)
                attention_masks = torch.cat(attention_masks, dim=0)
                outputs = bert_model(input_ids.to(device), attention_masks.to(device))

            if embed_type == 1:
                hidden_states = outputs[2]
                token_embeddings = torch.stack(hidden_states[-4:], dim=0)
                token_embeddings = token_embeddings.permute(1, 2, 0, 3)
                token_embeddings = token_embeddings.mean(axis=2)
            elif embed_type == 2:
                token_embeddings = outputs.pooler_output  # 不适合很多空值的情况,
            elif embed_type == 3:
                token_embeddings = outputs.last_hidden_state[0][0]  # [CLS] tokens

            output_embedding = token_embeddings.cpu().numpy()

        return input_ids, output_embedding, attention_masks


def model_eval(model, dataloader, num_labels, finetune=False, class_weight=None, task='eval'):
    '''only move the data into GPU when training and validating'''

    device = glovar.device_type
    model.eval()
    tokenized_texts = []
    logit_preds = []
    true_labels = []
    pred_labels = []
    # total_eval_accuracy = 0
    total_eval_loss = 0
    # nb_eval_steps = 0

    if class_weight is not None:
        pos_weight = torch.tensor(class_weight).to(device)
        criterion = CrossEntropyLoss(pos_weight=pos_weight)
    else:
        criterion = CrossEntropyLoss()

    for batch in dataloader:
        with torch.no_grad():
            b_labels = batch[2].to(device)  # [2]: labels
            if finetune:
                b_input_ids = batch[1].to(device)  # [1]: input sentence ids
                b_attentions = batch[3].to(device)  # [3]: att
                b_logits = model(b_input_ids, b_attentions)
            else:
                b_input_encoding = batch[0].to(device)  # [0]: input bert encoding features
                b_logits = model(b_input_encoding)

            b_prob = softmax(b_logits)
            # val_loss = criterion(b_prob.view(-1, num_labels), b_labels.type_as(b_prob).view(-1, num_labels))  # convert labels to float for calculation
            val_loss = criterion(b_prob, b_labels.type_as(b_prob))  # convert labels to float for calculation
            total_eval_loss += val_loss.item()

            # # save result
            b_labels = b_labels.to('cpu').numpy()
            b_prob = b_prob.to('cpu').numpy()
            pred_label = b_prob

            logit_preds.append(b_logits)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)
            tokenized_texts.append(b_input_ids)

    # Flatten outputs
    pred_labels = np.vstack(pred_labels)
    true_labels = np.vstack(true_labels)

    # print('pred_labels', pred_labels.shape)
    # print('true_labels', true_labels.shape)
    avg_val_loss = total_eval_loss / len(dataloader)

    pred_sparse = np.where(pred_labels > 0.5, 1, 0) if num_labels == 2 else np.argmax(pred_labels, axis=1)
    true_sparse = np.where(true_labels > 0.5, 1, 0) if num_labels == 2 else np.argmax(true_labels, axis=1)
    # print(true_sparse.shape, pred_sparse.shape)

    ## metrics
    auc_score = roc_auc_score(true_labels, pred_labels, multi_class='ovr')
    precison = precision_score(true_sparse, pred_sparse, average='macro')
    recall = recall_score(true_sparse, pred_sparse, average='macro')
    acc = accuracy_score(true_sparse, pred_sparse)
    f1 = f1_score(true_sparse, pred_sparse, average='macro')
    # print("AUC Score : %f" % auc_score)

    return tokenized_texts, pred_labels, true_labels, avg_val_loss, auc_score, precison, recall, acc, f1


def train_multi_label_model(model, num_labels, label_cols, train_dataloader, validation_dataloader, finetune=False, optimizer=None, scheduler=None, epochs=10, class_weight=None, patience=3, model_path='bert_clf.pt', verbose=0):
    """
    Below is our training loop. There's a lot going on, but fundamentally for each pass in our loop we have a trianing phase and a validation phase. At each pass we need to:

    Training loop:
    - Unpack our data inputs and labels
    - Load data onto the GPU for acceleration
    - Clear out the gradients calculated in the previous pass.
        - In pytorch the gradients accumulate by default (useful for things like RNNs) unless you explicitly clear them out.
    - Forward pass (feed input data through the network)
    - Backward pass (backpropagation)
    - Tell the network to update parameters with optimizer.step()
    - Track variables for monitoring progress

    Evalution loop:
    - Unpack our data inputs and labels
    - Load data onto the GPU for acceleration
    - Forward pass (feed input data through the network)
    - Compute loss on our validation data and track variables for monitoring progress
    The loss function is different from multi-label classifer

    Parameters:

    * model: model defined
    *   num_labels: number of labels
    *   label_cols: label names
    *   train_dataloader: train data loader
    *   validation_dataloader: validation data loader
    *   optimizer: optimizer. default is Adam
    *   scheduler: adjust learning rate dynamically; default is None.
    *   epochs: number of epochs
    """

    device = glovar.device_type
    print(device)
    # seed_val = 42
    # # threshold = 0.5

    # random.seed(seed_val)
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)

    training_stats = []
    best_score = 0
    best_epoch = 0
    best_model = copy.deepcopy(model.state_dict())
    cnt = 0

    total_t0 = time.time()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())

    if not os.path.exists(model_path[0:model_path.rfind("/")]):
        os.makedirs(model_path[0:model_path.rfind("/")])

    if class_weight is not None:
        pos_weight = torch.tensor(class_weight).to(device)
        criterion = CrossEntropyLoss(pos_weight=pos_weight)
    else:
        criterion = CrossEntropyLoss()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        t0 = time.time()
        total_train_loss = 0
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                if verbose > 1:
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            model.zero_grad()
            b_labels = batch[2].to(device)  # [2]: labels
            if finetune:
                b_input_ids = batch[1].to(device)  # [1]: input sentence ids
                b_attentions = batch[3].to(device)  # [3]: att
                b_logits = model(b_input_ids, b_attentions)
            else:
                b_input_encoding = batch[0].to(device)  # [0]: input bert encoding features
                b_logits = model(b_input_encoding)

            b_prob = softmax(b_logits)
            loss = criterion(b_prob, b_labels.type_as(b_prob))  # convert labels to float for calculation
            # loss = criterion(b_prob.view(-1, num_labels), b_labels.type_as(b_prob).view(-1, num_labels))  # convert labels to float for calculation

            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 可以试着删除

            clip_value = False
            if clip_value:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)

            clip_norm = False
            if clip_norm:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)  # MAML 不允许 clipnorm?! clip_coef = max_norm / (total_norm + 1e-6) ; p.grad.detach().mul_(clip_coef.to(p.grad.device)) # 为什么没起作用
            else:
                total_norm = 'No applied'

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        training_time = format_time(time.time() - t0)

        # print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Collection train and validation information
        # ========================================
        # print("Performance on training...")
        model.eval()
        tokenized_texts, pred_labels, true_labels, avg_train_loss, auc_score, precison, recall, train_acc, f1 = model_eval(model, train_dataloader, num_labels, finetune=finetune, class_weight=None)

        print("    Epoch {0}\t Train Loss: {1:.4f}\t Train Acc: {2:.4f}\t Train F1: {3:.4f}\t Train ovr AUC: {4:.4f}\t Train precision: {5:.4f}\t Train recall: {6:.4f}".format(epoch_i + 1, avg_train_loss, train_acc, f1, auc_score, precison, recall))

        # print("Running Validation...")
        t0 = time.time()
        model.eval()
        tokenized_texts, pred_labels, true_labels, avg_val_loss, auc_score, precison, recall, acc, f1 = model_eval(model, validation_dataloader, num_labels, finetune=finetune, class_weight=None)
        print("    Epoch {0}\t Val Loss: {1:.4f}\t Val Acc: {2:.4f}\t Val F1: {3:.4f}\t Val ovr AUC: {4:.4f}\t Val precision: {5:.4f}\t Val recall: {6:.4f}".format(epoch_i + 1, avg_val_loss, acc, f1, auc_score, precison, recall))

        validation_time = format_time(time.time() - t0)
        # Record all statistics from this epoch.
        training_stats.append({'epoch': epoch_i + 1, 'train_loss': avg_train_loss, 'train_acc': train_acc, 'val_loss': avg_val_loss, 'val_acc': acc, 'val F1 macro': f1, 'val ovr AUC': auc_score, 'val precison': precison, 'val recall': recall, 'Best epoch': best_epoch, 'Training Time': training_time, 'Validation Time': validation_time})

        # ========================================
        #               Early stop
        # ========================================

        if auc_score > best_score:
            best_score = auc_score
            best_epoch = epoch_i + 1
            best_model = copy.deepcopy(model.state_dict())
            print("model saved")
            cnt = 0
        else:
            cnt += 1
            if cnt == patience:
                print("\n")
                print("early stopping at epoch {0}".format(epoch_i + 1))
                break

    print("")
    print("Training complete!")
    torch.save(best_model, model_path)
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

    return model, training_stats, pred_labels, true_labels


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def device_checking(cpu_pref=False):
    # CUDA_LAUNCH_BLOCKING = 1

    # If there's a GPU available...
    if torch.cuda.is_available() and (not cpu_pref):

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def re_softmax(y, axis=1):
    """Compute softmax values for each sets of scores in x."""
    x = np.log(y / (1 - y))  # inverse of sigmoid to logits
    if axis == 0:
        return np.exp(x) / np.sum(np.exp(x), axis=0)  # logits to softmax
    else:
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)
