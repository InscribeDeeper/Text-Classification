# Bert 处理代码框架
1. 数据预处理
    - 只 focus 在 surprise  + Tokenizer特殊处理 # @ 
    - 在 imbalance 的数据集上尝试了一下. 然后发现不太行. 虽然 acc 很高, 但是不太行
    - 重新组装数据, 把不同 source 的数据集 combined 了一下, 用于训练 model
    - predict 并且保存了 这个 text
2. tokenization
3. 转换为sequence
4. 建立model
    * 调用预处理数据
    * 调用model
    * 调用BERT + Bi-LSTM
    * 输入model
    * 保存model
5. model evaluation
    * 读取model参数
    * evaluate
    * 进行另外一个数据的预处理
    * 重新进行evaluate


`!pip install transformers==2.5.1`

- lstm_cnn_mix_emotion
- lstm_cnn_mix_emotion_filtered
    - 删除30个char以下的句子
    - 删除最后一个标点符号 ?[?.!] => ? 只保留最后一个问号. 
    - 算一下AUC
- lstm_cnn_mix_emotion_filtered_v2
    - 在前一个基础上 重新运行了一遍, 没改变什么东西
- lstm_cnn_mix_emotion_filtered_v3
    - 训练前, 先shuffle 了一遍dataset (因为顺序会影响 model 的 前行方向)
    - v3 是最好的!!
- lstm_cnn_mix_emotion_filtered_v4
    - 和v3一样
    - 但是AUC Score : 0.961394 Epoch 135	 Train Loss: 0.2939	 Val Loss 0.2571	 Val Acc: 0.8908	 Val F1: 89.0819
- lstm_cnn_mix_emotion_filtered_v5
    - AUC Score : 0.963690 Epoch 227	 Train Loss: 0.2459	 Val Loss 0.2497	 Val Acc: 0.8883	 Val F1: 88.8337
- lstm_cnn_mix_emotion_filtered_v6
    - learning_rate = 1e-5 (从原来的1e-6)
- lstm_cnn_mix_emotion_filtered_v7
    - learning_rate = 1e-7 (从原来的1e-6)
    - AUC Score : 0.893547 Epoch 84	 Train Loss: 0.6632	 Val Loss 0.6633	 Val Acc: 0.8288	 Val F1: 82.8784
- lstm_cnn_mix_emotion_filtered_v8
    - learning_rate = 2e-7 (从原来的1e-6)
    - AUC Score : 0.898990 Epoch 154	 Train Loss: 0.5379	 Val Loss 0.5422	 Val Acc: 0.8263	 Val F1: 82.6303





# metrics
- agg on paragraph level
    - `out['pred_max'] = z.apply(lambda x: max(x))`
    - `out['pred_threshold0.62_sum'] = z.apply(lambda x: sum(np.array(x)> 0.62))`
    - `out['pred_mean']  = z.apply(lambda x: np.mean(x))`
    - `out['pred_threshold0.5_sum'] = z.apply(lambda x: sum(np.array(x)>0.5))`
    - `out['pred_prob_sum'] = z.apply(lambda x: sum(x))`
- agg on CONF call level
    - Average of all question probabilities
    - Top K (K=5, 10, …) questions by probability and then take the average
    - Proportion of questions with probability > threshold (e.g. 0.5). You can find the right threshold by histogram of probability
    - Minimum of maximum probability
- generate summary statistic



