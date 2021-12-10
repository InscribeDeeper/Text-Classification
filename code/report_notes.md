- 不同预处理对结果影响很小, 因为TFIDF考虑的是BOW
    - 但是如果针对 考虑顺序的model的话, 这里就会非常影响. 因为句子没分开, 句意就会乱. 
    - 最完整的 确实能提升 2% 的结果
    - Stemming 的影响有多大, 是否应该只考虑词库中的单词, 如果是这样, 影响有多大
    - word based 是否可以!? 随着时间会变化. 
    
- Kmeans + Majority votes
    - Cosine
    - L2 
    - 结果不大行

- LDA
    - topic modeling
    - 完全不working, 
        - 原因解释


- LR
    - 可以并行


    
- RF
    - 结果怎样
    - XGBT 能否用




- NB 
    - 结果
        - 最优 83% acc, 81 macro avg
    - 速度考虑
        - 不允许负数, 所以不能用降维
        - 相对 SVC 比较慢
    - 

- SVM
    - 结果  
        - 直接SVC都有86% acc 作为baseline, 85 macro avg
    - 原理
        - linear SVC is ovr - 适合多分类, 因为是线性kernel, minimize squared hinge loss
        - SVC is one over one - 适合二分类, 因为是非线性kernel, minimize hinge loss
    - GS 得到结果, binary = True, 并且  min_df 为1.
        - 原因解释: 
    - 速度考虑
        - 可以降维 但是不需要 因为是 linearSVC, 对应的原理优化
        - 而且速度很快, 
    - 稳定性 看CV结果
        - 有 92 cv f1 macro




- NN
    - 很明显的是, 这个数据集太简单了
    - 从training set上, 已经达到了100%, 这意味着 数据里面已经没有包含更多信息 可以被model 学习了
        - 虽然也是过拟合, 但是已经无法提高了
    - 同时 validation set也已经达到了瓶颈

- CNN 还行
    - textCNN 有上限

- RNNs太慢了, 不适合这个case, 但是可能会有所改善

- BERT
    - 核心是需要将文本分离, 那样的话, 重要信息 才能被 capture, 因为bert的参数设置的是100, 只能前100个单词被捕捉
    - 那这里就突显出来分离的必要性了.
    - 基于TFIDF 或者 CNN 都是BOW 不考虑顺序的 
    - 但是 BERT 是考虑句子意思的, 所以必须要分离
    - 因为考虑 信息不充分, 所以继续从原有数据中 提取特征, 选择 比较多的
    - 因为BERT 在 PYTORCH 支持更好, 所以BERT 相关的model都在pytorch 展开
    - 考虑 distill BERT? 
    - 这里没有 finetune, 只用了 pretrain, 因为太慢了. 
    - 所以只用 pretrain 的BERT 作为 extractor, 提取信息, 然后用 DENSE 去获取规律
        - 这里要深, 而且先不用 dropout, 而且 用 非线性 
    - RELU 因为参数过少 容易under fit, 虽然速度快, 比较快反映signal
    - 不 pretrain 效果会差很多


- 整合所有数据为 html


- 后续改进
    - 使用 成熟的工具 像: Vowpal Wabbit, interactive machine learning library. 
    - 特征工程, 怎么将更多的信息 非线性的 组合起来, 并且让model找到
        - TFIDF的信息已经被用完了 准确率仍然不够
        - 现在并不是复杂度不够, 需要降低复杂度, 做特征工程 降低维度!! 而是 信息太少了, 噪音太多了. 
        - 目前试用了 subject, 之后可以提取 from to 之类的 social network 的信息
    - rule based, 直接收集对应的词库




