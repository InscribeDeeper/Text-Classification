# Summary
After exploration on machine learning models, probability topic modeling, and deep learning models, this project make a comparison between 10 different models. The best preprocessing procedure are also explored for better performance on each model. With exploring optional setting related to data augmentation, up-sampling for small samples, lemmatization, stemming, and feature selection, this project use grid search, cross validation to find the best setting and preprocessing path for machine learning models. In addition, the clustering, topic modeling, Deep Neural Networks and multi-channel text CNN are implemented and explored in this project. However, the performance is not that acceptable in such models. At last, this project also explores the performance on finetune and pretrain BERT stacked with dense neural networks. The best performance has 0.84 macro F1 average score, which is obtained by both SVM and finetune BERT with a specific preprocessed input. 
In the future experiment on similar dataset, considering the exploration cost, SVM should be tested at first to know the baseline of model’s performance in the dataset. It is well implemented by Sklearn and can be quick trained with sparse matrix form input, which will reduce the risk of OOM as well. If the further exploration needed, the BERT related model should be explored. With more detail hyper-parameters tuning tricks and experience, the BERT model could have even better performance than 84% in this dataset. Furthermore, the finetuned model can also be transferred in other similar dataset to finish the classification task. For future reference, all the utils are wrapped and upload into GitHub  repository.
















