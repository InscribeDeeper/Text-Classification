- 不同预处理对结果影响很小, 因为TFIDF考虑的是BOW
    - 但是如果针对 考虑顺序的model的话, 这里就会非常影响. 因为句子没分开, 句意就会乱. 
    - 最完整的 确实能提升 2% 的结果
    
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


- 整合所有数据为 html


- 这是啥意思? sub class?
Micro average (averaging the total true positives, false negatives and
    false positives) is only shown for multi-label or multi-class
    with a subset of classes