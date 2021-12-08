from nltk.cluster import KMeansClusterer, cosine_distance
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_curve, auc, precision_recall_curve

from nltk.corpus import stopwords
# nltk.download('stopwords')
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def count_vectorizer(train_text, test_text, min_df=3, max_df=0.95):
    en_stopwords = stopwords.words('english')
    tfidf_vect = CountVectorizer(stop_words=en_stopwords, token_pattern=r'\b\w[\']?\w*\b', min_df=min_df, max_df=max_df)
    dtm_train = tfidf_vect.fit_transform(train_text)
    dtm_test = tfidf_vect.transform(test_text)

    word_to_idx = tfidf_vect.vocabulary_
    print("num of words:", len(word_to_idx))
    return dtm_train, dtm_test, word_to_idx, tfidf_vect


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
            clusterer = KMeansClusterer(num_clusters, cosine_distance, normalise=False, repeats=repeats, avoid_empty_clusters=True)
            train_cluster_pred = clusterer.cluster(dtm_train, assign_clusters=True)

    elif model == 'GMM':
        pass
        # GMM model not good in such case
        # clusterer = mixture.GaussianMixture(n_components=num_clusters, n_init=repeats, covariance_type='diag')
        # clusterer.fit(dtm_train)
        # train_cluster_pred = clusterer.predict(dtm_train)

    # Maping clusters into labels
    df = pd.DataFrame(list(zip(train_label, train_cluster_pred)), columns=['actual_class', 'cluster'])
    confusion = pd.crosstab(index=df.cluster, columns=df.actual_class)
    clusters_to_labels = confusion.idxmax(axis=1)

    print("Cluster to label mapping: ")
    for idx, t in enumerate(clusters_to_labels):
        print("Cluster {} <-> label {}".format(idx, t))
    print("\n")

    return clusterer, clusters_to_labels


def pred_clustering_model(dtm_test, clusterer, clusters_to_labels):
    test_cluster_pred = [clusterer.classify(v) for v in dtm_test]
    predict = [clusters_to_labels[i] for i in test_cluster_pred]
    return predict
