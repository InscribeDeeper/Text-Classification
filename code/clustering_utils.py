from nltk.cluster import KMeansClusterer, cosine_distance  # will get nan when u v are zero?
import pandas as pd
from sklearn.cluster import KMeans
from gensim.utils import tokenize
import pyLDAvis
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import pandas as pd
import numpy as np

################################################
## Majority vote rules
################################################


def link_group_to_label(train_label, train_pred, num_topics=100):
    """with majority vote rule"""
    # Maping clusters into labels
    df = pd.DataFrame(list(zip(train_label, train_pred)), columns=['actual_class', 'cluster'])
    confusion = pd.crosstab(index=df.cluster, columns=df.actual_class)

    ## handle no group
    full = pd.DataFrame(index=range(num_topics), columns=train_label.unique())
    full.loc[:, 'no_group'] = 0.1  # the minimum is 1
    merge_full = full.combine_first(confusion).fillna(0)
    group_to_label = merge_full.idxmax(axis=1)

    ## print out mapping
    print("Group to label mapping: ")
    for idx, t in enumerate(group_to_label):
        print("Group {} <-> label {}".format(idx, t))
    print("\n")
    return group_to_label


################################################
## Clustering tools
################################################


def fit_clustering_model(dtm_train, train_label, num_clusters, metric='Cosine', model='KMeans', repeats=20):
    '''

    '''
    assert metric in ['Cosine', 'L2']
    assert model in ['KMeans']

    # model training
    if model == 'KMeans':
        if metric == 'Cosine':
            # normalise should be true!
            clusterer = KMeansClusterer(num_clusters, cosine_distance, normalise=True, repeats=repeats, avoid_empty_clusters=True)
            train_cluster_pred = clusterer.cluster(dtm_train, assign_clusters=True)
        elif metric == 'L2':
            clusterer = KMeans(n_clusters=num_clusters, n_init=repeats).fit(dtm_train)
            train_cluster_pred = clusterer.labels_.tolist()

    elif model == 'GMM':
        pass
        # GMM model not good in such case
        # clusterer = mixture.GaussianMixture(n_components=num_clusters, n_init=repeats, covariance_type='diag')
        # clusterer.fit(dtm_train)
        # train_cluster_pred = clusterer.predict(dtm_train)

    # Maping clusters into labels
    clusters_to_labels = link_group_to_label(train_label, train_cluster_pred, num_clusters)
    return clusterer, clusters_to_labels


def pred_clustering_model(dtm_test, clusterer, clusters_to_labels):
    try:
        test_cluster_pred = clusterer.predict(dtm_test)  # for sklearn clustering with L2
    except Exception:
        test_cluster_pred = [clusterer.classify(v) for v in dtm_test]  # for nltk clustering with Cosine similiarity
    predict = [clusters_to_labels[i] for i in test_cluster_pred]
    return predict


################################################
## Topic modeling tools
################################################


def transform_lda_corpus(docs, vocabulary=None):
    assert isinstance(docs, pd.Series)
    idx_to_word = vocabulary

    tokenized_docs = docs.apply(lambda x: list(tokenize(x))).to_list()
    if idx_to_word is None:
        idx_to_word = Dictionary(tokenized_docs)
    sparse_corpus = [idx_to_word.doc2bow(doc) for doc in tokenized_docs]
    return idx_to_word, sparse_corpus


def fit_topic_model(docs, num_topics=100, save_name='lda_gensim_model'):
    '''
    docs is the pd Series
    output lda model and topic prediction on docs
    '''
    vocabulary, sparse_corpus = transform_lda_corpus(docs, vocabulary=None)
    lda = LdaModel(sparse_corpus, num_topics=num_topics, minimum_probability=0.0001, dtype=np.float64)
    if save_name is not None:
        lda.save(save_name)
        lda = LdaModel.load(save_name)  # index 会变小吗
    return lda, vocabulary


def pred_topic_model(lda, docs, vocabulary):
    assert vocabulary is not None
    _, sparse_corpus = transform_lda_corpus(docs, vocabulary=vocabulary)
    pred = lda[sparse_corpus]

    topic_distribution = lil_to_dataframe(pred, nrows=len(docs), ncols=lda.num_topics)

    ## checking for no topic
    a = topic_distribution.sum(axis=1)
    print(a[a == 0])

    pred = topic_distribution.idxmax(axis=1, skipna=False)
    return pred, topic_distribution


def lil_to_dataframe(pred, nrows, ncols):
    res = {}
    for row, doc_topics in enumerate(pred):
        res[row] = dict(doc_topics)

    d1 = pd.DataFrame(index=range(nrows), columns=range(ncols))
    d2 = pd.DataFrame.from_dict(res, orient='index')
    # d3 = d1.combine_first(d2)
    d3 = d1.combine_first(d2).fillna(0)
    return d3


def visualize_LDA_model(docs, voc, lda):
    _, sparse_corpus = transform_lda_corpus(docs, vocabulary=voc)
    pyLDAvis.enable_notebook()
    panel = pyLDAvis.gensim.prepare(lda, corpus=sparse_corpus, dictionary=voc, mds='tsne')
    return panel


def load_gensim_LDA_model(save_name='lda_gensim_model'):
    return LdaModel.load(save_name)  # key 会少一个
