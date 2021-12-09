import numpy as np
import sys
import datetime
import gc
import pynvml
import psutil
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report
from random import sample
import learn2learn as l2l
import time
from tensorboardX import SummaryWriter


class sample_col:
    def __init__(self, train_cols, valid_cols):
        self.train_cols = train_cols
        self.valid_cols = valid_cols

    def sample_train(self):
        return sample(self.train_cols, 1)[0]

    def sample_valid(self):
        return sample(self.valid_cols, 1)[0]


def accuracy(outputs, true):
    # outputs = outputs.round()
    outputs = (outputs >= 0.5) * 1
    # print(outputs)
    return (outputs == true).sum().float() / true.size(0)


def fast_adapt(support_data_x, support_data_y, query_data_x, query_data_y, learner, criterion, adaptation_steps, device, return_record=False, eval_mode=False, show_classification_report=False):

    # generate label for support data
    # num_support_obs = support_data_x.size(0)
    # print(num_support_obs, support_data_x.size())
    # support_data_y = torch.tensor(([0.0] * int(num_support_obs / 2)+ [1.0] * int(num_support_obs / 2)))

    # shuffle the support data
    # print("support_data_x.dtype", support_data_x.dtype)
    # print("support_data_y.dtype", support_data_y.dtype)

    idx = torch.randperm(support_data_x.size(0))
    support_data_x = support_data_x[idx]
    support_data_y = support_data_y[idx]

    # move all data to device
    support_data_x = support_data_x.to(device)
    support_data_y = support_data_y.to(device)
    query_data_x = query_data_x.to(device)
    query_data_y = query_data_y.to(device)

    # adapt record
    adapt_record = {'acc': [], 'loss': []}

    # adapt model
    for i in range(adaptation_steps):
        learner.train()
        # with torch.cuda.amp.autocast():
        with torch.backends.cudnn.flags(enabled=False):  # magic code
            outputs = learner(support_data_x).view(-1)
            loss = criterion(outputs, support_data_y)
            # learner.adapt(loss, allow_nograd=True, allow_unused=True)
            try:
                learner.adapt(loss)
            except Exception:  # For gradient explosion NAN
                torch.nn.utils.clip_grad_value_(learner.parameters(), clip_value=1)
                learner.adapt(loss, allow_nograd=True, allow_unused=True)

        adapt_record['acc'].append(accuracy(outputs, support_data_y).item())
        adapt_record['loss'].append(loss.item())
        if not eval_mode:
            print("      adaptation_steps", i + 1, " (support) loss: ", loss.item())

    # meta loss and acc:
    if not eval_mode:
        learner.train()
    else:
        learner.eval()
    with torch.backends.cudnn.flags(enabled=False):  # magic code
        outputs = learner(query_data_x).view(-1)
    meta_loss = criterion(outputs, query_data_y)
    meta_acc = accuracy(outputs, query_data_y)

    if show_classification_report:
        with torch.no_grad():
            print(classification_report(query_data_y.cpu().numpy(), outputs.cpu().numpy().round()))

    # print("deleted vars")
    # del support_data_x
    # del support_data_y
    # del query_data_x
    # del query_data_y

    # print("deleted vars by empty_cache")
    # torch.cuda.empty_cache()

    if not return_record:
        return learner, meta_loss, meta_acc
    else:
        return learner, meta_loss, meta_acc, {'acc': adapt_record['acc'][-1], 'loss': adapt_record['loss'][-1]}


# train function
def MAML_train(base_model, train_loaders, valid_loaders, learning_rate_alpha, learning_rate_beta, n_iterations, meta_batch_size, device, dtype=torch.float16, first_order=False, save_meta=False, model_path=None, verbose=True, adaptation_steps=2, max_len=100):
    log_path = "log_" + model_path
    writer = SummaryWriter(log_dir=log_path)

    # maml wrapper
    if dtype == torch.float16:
        # float32 是有必要的, 特别是在 函数接近 0 的时候, 区别会非常的明显
        base_model = base_model.half().to(device)  # 这个需要修改optimizer 优化算法，Adam我在使用过程中，在某些参数的梯度为0的时候，更新权重后，梯度为零的权重变成了NAN，这非常奇怪，但是Adam算法对于全精度数据类型却没有这个问题。
    else:
        base_model = base_model.to(device)

    maml = l2l.algorithms.MAML(base_model, lr=learning_rate_alpha, first_order=first_order)

    # build optimizer & loss function
    # Currently not all learners support FP16. This change is not trivia because smoothed gradients need to be in FP32 while gradients are in FP16.
    # Besides, parameters' master copy needs to be in FP32 for accumulation during model update by learners. It got converted to FP16 before forward.

    if dtype == torch.float16:
        optimizer = torch.optim.SGD(maml.parameters(), lr=learning_rate_beta)
    else:
        optimizer = torch.optim.Adam(maml.parameters(), lr=learning_rate_beta)  # optimizer.step() 之后可能变成了NAN

    # criterion = torch.nn.BCELoss(reduction='mean') # for pre-calc sigmoid output
    criterion = torch.nn.BCEWithLogitsLoss()
    # amp
    # maml, optimizer = amp.initialize(maml, optimizer, opt_level='O1')

    # access keys
    train_keys = train_loaders.keys()
    valid_keys = valid_loaders.keys()

    # initilize sampler
    sampler = sample_col(train_keys, valid_keys)

    # record
    history = {'train_acc': [], 'train_loss': [], 'valid_acc': [], 'valid_loss': [], 'train_adapt_record': [], 'valid_adapt_record': []}

    # train and validation
    for iteration in range(n_iterations):
        print("\n\niteration: ", iteration + 1)
        mem_ratio()
        # time
        start_time = time.time()
        # meta
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        # adaptation record
        the_train_adapt_record = dict.fromkeys(train_keys)
        for cur_key in train_keys:
            the_train_adapt_record[cur_key] = {'acc': 0, 'loss': 0}
        the_valid_adapt_record = dict.fromkeys(valid_keys)
        for cur_key in valid_keys:
            the_valid_adapt_record[cur_key] = {'acc': 0, 'loss': 0}
        # counter
        the_train_counter = dict.fromkeys(train_keys)
        for cur_key in train_keys:
            the_train_counter[cur_key] = 0
        the_valid_counter = dict.fromkeys(valid_keys)
        for cur_key in valid_keys:
            the_valid_counter[cur_key] = 0

        for j in range(meta_batch_size):

            # ######################
            # train
            # ######################
            learner = maml.clone()
            cur_train_key = sampler.sample_train()
            the_train_counter[cur_train_key] += 1
            # data
            cur_x, cur_y = next(train_loaders[cur_train_key])
            cur_support_x = torch.tensor(cur_x[0].reshape(-1, max_len, 768), dtype=dtype)
            cur_query_x = torch.tensor(cur_x[1], dtype=dtype)
            cur_support_y = torch.tensor(cur_y[0].reshape(-1), dtype=dtype)  # if not announce the dtype, tensor will automatically assign int type
            cur_query_y = torch.tensor(cur_y[1], dtype=dtype)
            # adapt

            # train_adapt_record saves performance before parameter changes
            learner, meta_loss, meta_acc, train_adapt_record = fast_adapt(cur_support_x, cur_support_y, cur_query_x, cur_query_y, learner=learner, criterion=criterion, adaptation_steps=adaptation_steps, device=device, return_record=True)
            # backward
            for name, param in learner.named_parameters():
                writer.add_histogram(name + '_meta_learner', param.clone().cpu().data.numpy(), (iteration + 1) * 100 + j + 1)

            meta_loss.backward()
            meta_train_error += meta_loss.item()
            meta_train_accuracy += meta_acc.item()
            print("  meta_batch: ", j + 1, ' ==> (query) meta_loss: ', meta_loss.item())

            # record
            the_train_adapt_record[cur_train_key]['acc'] += train_adapt_record['acc']
            the_train_adapt_record[cur_train_key]['loss'] += train_adapt_record['loss']

            # the_train_adapt_record[cur_train_key]['acc'].append(train_adapt_record['acc'])
            # the_train_adapt_record[cur_train_key]['loss'].append(train_adapt_record['loss'])

            # ######################
            # validation -> eval mode = will not accumulate grad
            # ######################
            learner = maml.clone()
            cur_valid_key = sampler.sample_valid()
            the_valid_counter[cur_valid_key] += 1
            # data
            cur_x, cur_y = next(train_loaders[cur_train_key])
            cur_support_x = torch.tensor(cur_x[0].reshape(-1, max_len, 768), dtype=dtype)
            cur_query_x = torch.tensor(cur_x[1], dtype=dtype)
            cur_support_y = torch.tensor(cur_y[0].reshape(-1), dtype=dtype)  # if not announce the dtype, tensor will automatically assign int type
            cur_query_y = torch.tensor(cur_y[1], dtype=dtype)
            # adapt

            if j > meta_batch_size - 3:  # print classification report for the last 2 round
                print("\n========== " + cur_valid_key, "=============\n")
                learner, meta_loss, meta_acc, valid_adapt_record = fast_adapt(cur_support_x, cur_support_y, cur_query_x, cur_query_y, learner=learner, criterion=criterion, adaptation_steps=adaptation_steps, device=device, return_record=True, eval_mode=True, show_classification_report=True)
                torch.save(learner.module.state_dict(), model_path + "_" + cur_valid_key)  # save the adapted model for each datat source
            else:
                learner, meta_loss, meta_acc, valid_adapt_record = fast_adapt(cur_support_x, cur_support_y, cur_query_x, cur_query_y, learner=learner, criterion=criterion, adaptation_steps=adaptation_steps, device=device, return_record=True, eval_mode=True, show_classification_report=False)

            meta_valid_error += meta_loss.item()
            meta_valid_accuracy += meta_acc.item()
            the_valid_adapt_record[cur_valid_key]['acc'] += valid_adapt_record['acc']
            the_valid_adapt_record[cur_valid_key]['loss'] += valid_adapt_record['loss']

            gpu_memory_usage()

            # torch.cuda.empty_cache()
            # meta batch training

        # grad normalization # 因为在累加, 在这下面在 进行 optimizer.step(). 所以这里要进行一种norm. 可不可以直接用 clip?

        for name, param in maml.named_parameters():
            writer.add_histogram(name + '_before_norm_clip', param.clone().cpu().data.numpy(), iteration)

        print("*" * 50)
        print("checking clipping influence")

        #  grad_batch_mean = True
        for p in maml.parameters():  # meta batch is the sampling from different domains
            p.grad.data.mul_(1.0 / meta_batch_size)  # 这样可以缩小 grad, 就怕突然有一个 outlier 让 grad 直接爆炸了

        clip_value = True
        if clip_value:
            torch.nn.utils.clip_grad_value_(maml.parameters(), clip_value=1)

        clip_norm = False
        ## 这个 clip_norm 会影响那么大的吗? 会, 因为会放大  t * clip_norm / l2norm(t), L2 比1小的时候, 就会放大. 但是不需要.
        if clip_norm:
            total_norm = torch.nn.utils.clip_grad_norm_(maml.parameters(), max_norm=1)  # MAML 不允许 clipnorm?! clip_coef = max_norm / (total_norm + 1e-6) ; p.grad.detach().mul_(clip_coef.to(p.grad.device)) # 为什么没起作用
        else:
            total_norm = 'No applied'

        meta_max_grad = []
        for idx, p in enumerate(maml.parameters()):  # meta batch is the sampling from different domains
            max_in_layer = p.grad.data.clone().detach().abs().max().item()
            meta_max_grad.append([idx, max_in_layer])
        print("  grad data", meta_max_grad)
        print("  max grad data in layers", max([j for i, j in meta_max_grad]))
        print("  total_norm ", total_norm)

        for name, param in maml.named_parameters():
            value = param.clone().cpu().data.numpy()
            # if value is None:
            #     value = np.
            writer.add_histogram(name + '_after_norm_clip', value, iteration)

        # update params
        optimizer.step()  # optimizer 要允许 float16
        optimizer.zero_grad()

        # check if save:
        if save_meta and iteration == (n_iterations - 1):
            torch.save(maml.module.state_dict(), model_path)

        # record
        history['train_acc'].append(meta_train_accuracy / meta_batch_size)
        history['train_loss'].append(meta_train_error / meta_batch_size)
        history['valid_acc'].append(meta_valid_accuracy / meta_batch_size)
        history['valid_loss'].append(meta_valid_error / meta_batch_size)

        writer.add_scalar('Loss/train', meta_train_error / meta_batch_size, iteration)
        writer.add_scalar('Loss/valid', meta_valid_error / meta_batch_size, iteration)
        writer.add_scalar('Accuracy/train', meta_train_accuracy / meta_batch_size, iteration)
        writer.add_scalar('Accuracy/valid', meta_valid_accuracy / meta_batch_size, iteration)

        # train adapt record
        cur_train_record = dict.fromkeys(train_keys)
        for cur_key in train_keys:
            if the_train_counter[cur_key] != 0:
                the_record = {'acc': the_train_adapt_record[cur_key]['acc'] / the_train_counter[cur_key], 'loss': the_train_adapt_record[cur_key]['loss'] / the_train_counter[cur_key]}
            else:
                the_record = {'acc': None, 'loss': None}

            cur_train_record[cur_key] = the_record
        history['train_adapt_record'].append(cur_train_record)
        # valid adapt record
        cur_valid_record = dict.fromkeys(valid_keys)
        for cur_key in valid_keys:
            if the_valid_counter[cur_key] != 0:
                the_record = {'acc': the_valid_adapt_record[cur_key]['acc'] / the_valid_counter[cur_key], 'loss': the_valid_adapt_record[cur_key]['loss'] / the_valid_counter[cur_key]}
            else:
                the_record = {'acc': None, 'loss': None}
            cur_valid_record[cur_key] = the_record
        history['valid_adapt_record'].append(cur_valid_record)

        # Print some metrics
        if verbose:
            print('\n')
            print('Iteration', iteration)
            print('Time: ', time.time() - start_time)
            print('Meta Train Error', meta_train_error / meta_batch_size)
            print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size)
            print('Meta Valid Error', meta_valid_error / meta_batch_size)
            print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size)
            print('*' * 5 + 'Adaptation' + '*' * 5)
            print('Train:')
            for cur_key in train_keys:
                print(f'Class {cur_key}')
                print(f"train acc : {history['train_adapt_record'][-1][cur_key]['acc']}, train loss: {history['train_adapt_record'][-1][cur_key]['loss']}")
            print('Valid:')
            for cur_key in valid_keys:
                print(f'Class {cur_key}')
                print(f"valid acc : {history['valid_adapt_record'][-1][cur_key]['acc']}, valid loss: {history['valid_adapt_record'][-1][cur_key]['loss']}")

            print("\n\n" + '#' * 30)
            gpu_memory_log()
            print('#' * 30 + "\n\n")

    writer.export_scalars_to_json(log_path + "/all_scalars.json")
    writer.close()

    return history


def history_vis(history, train_cols, valid_cols):

    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['valid_loss'], label='Valid')

    plt.legend()
    plt.title('Meta Loss')
    plt.show()

    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['valid_acc'], label='Valid')
    plt.legend()
    plt.title('Meta Accuracy')
    plt.show()

    for cur_key in train_cols:
        cur_data = [cur_record[cur_key]['acc'] for cur_record in history['train_adapt_record']]
        plt.plot(cur_data, label=cur_key)
    plt.legend()
    plt.title('Adaptation Accuracy')
    plt.show()

    for cur_key in train_cols:
        cur_data = [cur_record[cur_key]['loss'] for cur_record in history['train_adapt_record']]
        plt.plot(cur_data, label=cur_key)
    plt.legend()
    plt.title('Adaptation Loss')
    plt.show()

    for cur_key in valid_cols:
        cur_data = [cur_record[cur_key]['acc'] for cur_record in history['valid_adapt_record']]
        plt.plot(cur_data, label=cur_key)
    plt.legend()
    plt.title('Adaptation Accuracy')
    plt.show()

    for cur_key in valid_cols:
        cur_data = [cur_record[cur_key]['loss'] for cur_record in history['valid_adapt_record']]
        plt.plot(cur_data, label=cur_key)
    plt.legend()
    plt.title('Adaptation Loss')
    plt.show()


def mem_ratio():
    mem = psutil.virtual_memory()
    used = str(round(mem.used / 1024 / 1024))
    use_per = str(round(mem.percent))
    mem = psutil.virtual_memory()
    print("RAM USED: " + used + "M(" + use_per + "%)")
    return None


def show_RAM(unit='KB', threshold=1):
    '''查看变量占用内存情况

    :param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
    :param threshold: 仅显示内存数值大于等于threshold的变量
    '''
    scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
    for i in list(globals().keys()):
        memory = eval("sys.getsizeof({})".format(i)) // scale
        if memory >= threshold:
            print(i, memory)
    return None
    # show_memory(unit='MB', threshold=1)


def gpu_memory_usage(device=0, verbose=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('\n')
    Used = "Used Memory:%.f Mb" % float(meminfo.used / 1024**2)
    Free = "Free Memory:%.f Mb" % float(meminfo.free / 1024**2)
    Total = "Total Memory:%.f Mb" % float(meminfo.total / 1024**2)
    # usage = "Used %f %% => Used Memory:%f Mb, Total Memory:%f Mb" % (100 * float(meminfo.used / 1024**2) / float(meminfo.total / 1024**2, ), float(meminfo.used / 1024**2), float(meminfo.total / 1024**2, ))
    usage = "GPU Used %.2f %% of Total Memory:%.f Mb" % (100 * float(meminfo.used / 1024**2) / float(meminfo.total / 1024**2, ), float(meminfo.total / 1024**2, ))
    print(Used) if verbose > 1 else None
    print(Total) if verbose > 2 else None
    print(Free) if verbose > 3 else None
    print(usage) if verbose >= 0 else None
    pynvml.nvmlShutdown()
    return None


# !/usr/bin/python
# -*- coding: utf-8 -*-
# ####################################
# File name : gpu_mem.py
# Create date : 2019-06-02 16:56
# Modified date : 2019-06-02 16:59
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
# ####################################


def _get_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            tensor = obj
        else:
            continue
        if tensor.is_cuda:
            yield tensor


def _write_log(f, write_str):
    print(write_str)
    f.write("%s\n" % write_str)


def gpu_memory_log(gpu_log_file="gpu_mem.log", device=0):
    stack_layer = 1
    func_name = sys._getframe(stack_layer).f_code.co_name
    file_name = sys._getframe(stack_layer).f_code.co_filename
    line = sys._getframe(stack_layer).f_lineno
    now_time = datetime.datetime.now()
    log_format = 'LINE:%s, FUNC:%s, FILE:%s, TIME:%s, CONTENT:%s'

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

    with open(gpu_log_file, 'a+') as f:
        write_str = log_format % (line, func_name, file_name, now_time, "")
        _write_log(f, write_str)

        ts_list = [tensor.size() for tensor in _get_tensors()]
        new_tensor_sizes = {(type(x), tuple(x.size()), ts_list.count(x.size()), np.prod(np.array(x.size())) * 4 / 1024**2) for x in _get_tensors()}
        for t, s, n, m in new_tensor_sizes:
            write_str = '[tensor: %s * Size:%s | Memory: %s M | %s]' % (str(n), str(s), str(m * n)[:6], str(t))
            _write_log(f, write_str)

        write_str = "memory_allocated:%f Mb" % float(torch.cuda.memory_allocated() / 1024**2)
        _write_log(f, write_str)
        write_str = "max_memory_allocated:%f Mb" % float(torch.cuda.max_memory_allocated() / 1024**2)
        _write_log(f, write_str)
        write_str = "memory_cached:%f Mb" % float(torch.cuda.memory_cached() / 1024**2)
        _write_log(f, write_str)
        write_str = "max_memory_cached:%f Mb" % float(torch.cuda.max_memory_cached() / 1024**2)
        _write_log(f, write_str)
        write_str = "Used Memory:%f Mb" % float(meminfo.used / 1024**2)
        _write_log(f, write_str)
        write_str = "Free Memory:%f Mb" % float(meminfo.free / 1024**2)
        _write_log(f, write_str)
        write_str = "Total Memory:%f Mb" % float(meminfo.total / 1024**2)
        _write_log(f, write_str)

    pynvml.nvmlShutdown()
    return None
