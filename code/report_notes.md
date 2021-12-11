
-   公开表现

-   不同预处理对结果影响很小, 因为 TFIDF 考虑的是 BOW
    -   但是如果针对 考虑顺序的 model 的话, 这里就会非常影响. 因为句子没分开, 句意就会乱.
    -   最完整的 确实能提升 2% 的结果
    -   Stemming 的影响有多大, 是否应该只考虑词库中的单词, 如果是这样, 影响有多大
    -   word based 是否可以!? 随着时间会变化.




- EDA
    -  数据的noise 太多
    - 用TF group by class 求和, 然后找到差异最大的那些, 但是 作为 DENSE 的input 还是无法改善太多
    - 已做
        - POS NLTK只提取名词
        - 是否能用 lexicon based : 看不同的class之间, 词汇非交集是什么

    - 未做: 
        - outliers 筛选, 根据long_string和 error message
    - 


- Feature engineering
    * 构建TFIDF-incrementally
        * 先只考虑 reply 看效果, 然后对比
        * 是不是TF-IDF可以用不同的权重求和
    - 可选项
        * Subject
        * from的 邮箱后缀
        * Lines
        * Organization
        * Summary
    - 提取entity + 名词 => 根据 这个matrix去聚类



-   Kmeans + Majority votes

    -   Cosine
    -   L2
    -   结果不大行
    -   小样本表现很好

-   LDA

    -   topic modeling
    -   完全不 working,
        -   原因解释

-   LR
    -   可以并行
-   RF

    -   结果怎样
    -   XGBT 能否用

-   NB

    -   结果
        -   最优 83% acc, 81 macro avg
    -   速度考虑
        -   不允许负数, 所以不能用降维
        -   相对 SVC 比较慢
    -

-   SVM

    -   结果
        -   直接 SVC 都有 86% acc 作为 baseline, 85 macro avg
        -   Hinge loss
    -   原理
        -   linear SVC is ovr - 适合多分类, 因为是线性 kernel, minimize squared hinge loss
        -   SVC is one over one - 适合二分类, 因为是非线性 kernel, minimize hinge loss
    -   GS 得到结果, binary = True, 并且 min_df 为 1.
        -   原因解释:
    -   速度考虑
        -   可以降维 但是不需要 因为是 linearSVC, 对应的原理优化
        -   而且速度很快,
    -   稳定性 看 CV 结果
        -   有 92 cv f1 macro

-   NN

    -   很明显的是, 这个数据集太简单了
    -   从 training set 上, 已经达到了 100%, 这意味着 数据里面已经没有包含更多信息 可以被 model 学习了
        -   虽然也是过拟合, 但是已经无法提高了
    -   同时 validation set 也已经达到了瓶颈

-   CNN 还行
    -   Gensim_w2v + textCNN
    -   textCNN 有上限

-   RNNs 太慢了, 不适合这个 case, 但是可能会有所改善

-   BERT
    -   pretrained_BERT_w2v  + CNNRNN   
    -   BERT FINETUNE 的上限
        -   原因
    -   核心是需要将文本分离, 那样的话, 重要信息 才能被 capture, 因为 bert 的参数设置的是 100, 只能前 100 个单词被捕捉
    -   那这里就突显出来分离的必要性了.
    -   基于 TFIDF 或者 CNN 都是 BOW 不考虑顺序的
    -   但是 BERT 是考虑句子意思的, 所以必须要分离
    -   因为考虑 信息不充分, 所以继续从原有数据中 提取特征, 选择 比较多的
    -   因为 BERT 在 PYTORCH 支持更好, 所以 BERT 相关的 model 都在 pytorch 展开
    -   考虑 distill BERT?
    -   这里没有 finetune, 只用了 pretrain, 因为太慢了.
    -   所以只用 pretrain 的 BERT 作为 extractor, 提取信息, 然后用 DENSE 去获取规律
        -   这里要深, 而且先不用 dropout, 而且 用 非线性
    -   RELU 因为参数过少 容易 under fit, 虽然速度快, 比较快反映 signal
    -   不 pretrain 效果会差很多
    -   The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks.
    -   缺点
        -   All sentences must be padded or truncated to a single, fixed length.
        -   The maximum sentence length is 512 tokens.

-   BertModel
-   BertForPreTraining
-   BertForMaskedLM
-   BertForNextSentencePrediction
-   BertForSequenceClassification - The one we’ll use.
-   BertForTokenClassification
-   BertForQuestionAnswering
-   DOC2VECTOR

-   整合所有数据为 html

-   后续改进
    -   使用 成熟的工具 像: Vowpal Wabbit, interactive machine learning library.
    -   特征工程, 怎么将更多的信息 非线性的 组合起来, 并且让 model 找到
        -   TFIDF 的信息已经被用完了 准确率仍然不够
        -   现在并不是复杂度不够, 需要降低复杂度, 做特征工程 降低维度!! 而是 信息太少了, 噪音太多了.
        -   目前试用了 subject, 之后可以提取 from to 之类的 social network 的信息
    -   rule based, 直接收集对应的词库
    -   可能 deploy in AWS if ML solution
    -   调用其他 robeta
    -   majority votes? 
    -   对于没有的 class, 可能需要提升一下weight, upsampling 或者改变权重都行. 
    -   BertGCN 这是graph
    
    

# 经验总结

-   如果在写代码之前, 能够清楚知道自己的 function 要完成什么功能, 那就能很好的把这些工具拼在一起. 而目前是不知道的, 在 exploration, 没有一个标准的 framework, 所以就会在原有的过程中 不断的修改.
    -   所以需要多看别人的 package, 然后学习他的 framework 构建包
    -   类似于 fit transform, fit predict 之类的
