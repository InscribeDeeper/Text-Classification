from nltk.cluster import KMeansClusterer, cosine_distance
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_curve, auc, precision_recall_curve

from gensim.utils import tokenize
import pyLDAvis
import pyLDAvis.gensim
from gensim.models import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
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


def count_vectorizer(train_text, test_text, min_df=3, max_df=0.95):
    en_stopwords = stopwords.words('english')
    count_vect = CountVectorizer(stop_words=en_stopwords, token_pattern=r'\b\w[\']?\w*\b', min_df=min_df, max_df=max_df)
    dtm_train = count_vect.fit_transform(train_text)
    dtm_test = count_vect.transform(test_text)

    word_to_idx = count_vect.vocabulary_
    print("num of words:", len(word_to_idx))
    return dtm_train, dtm_test, word_to_idx, count_vect


def tfidf_vectorizer(train_text, test_text, min_df=3, max_df=0.95):
    en_stopwords = stopwords.words('english')
    tfidf_vect = TfidfVectorizer(stop_words=en_stopwords, token_pattern=r'\b\w[\']?\w*\b', norm='l2', min_df=min_df, max_df=max_df)
    dtm_train = tfidf_vect.fit_transform(train_text)
    dtm_test = tfidf_vect.transform(test_text)

    word_to_idx = tfidf_vect.vocabulary_
    print("num of words:", len(word_to_idx))
    return dtm_train, dtm_test, word_to_idx, tfidf_vect


def dimension_reduction(dtm, method='svd', out_dim=200, verbose=0):
    transform_mapper = None

    if method == 'svd':
        print("Dimension reduction with truncate SVD:")
        print("   input columns with ", dtm.shape[1])
        print("   output columns with ", out_dim)

        transform_mapper = TruncatedSVD(n_components=out_dim)
        dtm = transform_mapper.fit_transform(dtm)
        if verbose > 0:
            print("singular_values_: ", transform_mapper.singular_values_)

    elif method == 'tsne':
        print("Notes: T-SNE is only for visualization, not preprocessing")
        assert out_dim < 3, "out_dim should less than 3 with t-sne, this is for visualization"
        # transform_mapper = TSNE(n_components=out_dim, metric='cosine')
        dtm = TSNE(n_components=out_dim).fit_transform(dtm)
    return dtm, transform_mapper


def fit_clustering_model(dtm_train, train_label, num_clusters, metric='Cosine', model='KMeans', repeats=20):
    '''

    '''
    assert metric in ['Cosine']
    assert model in ['KMeans']

    # model training
    if model == 'KMeans':
        if metric == 'Cosine':
            # normalise has to be False!
            clusterer = KMeansClusterer(num_clusters, cosine_distance, normalise=False, repeats=repeats, avoid_empty_clusters=True)
            train_cluster_pred = clusterer.cluster(dtm_train, assign_clusters=True)

    elif model == 'GMM':
        pass
        # GMM model not good in such case
        # clusterer = mixture.GaussianMixture(n_components=num_clusters, n_init=repeats, covariance_type='diag')
        # clusterer.fit(dtm_train)
        # train_cluster_pred = clusterer.predict(dtm_train)

    # Maping clusters into labels
    clusters_to_labels = link_group_to_label(train_label, train_cluster_pred)
    return clusterer, clusters_to_labels


def pred_clustering_model(dtm_test, clusterer, clusters_to_labels):
    test_cluster_pred = [clusterer.classify(v) for v in dtm_test]
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
        lda = LdaModel.load(save_name)
    return lda, vocabulary


def pred_topic_model(lda, docs, vocabulary=None):
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
    return LdaModel.load(save_name)
