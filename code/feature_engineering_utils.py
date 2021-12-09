from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


def count_vectorizer(train_text, test_text, stop_words=False, binary=False, min_df=3, max_df=0.95):
    en_stopwords = stopwords.words('english') if stop_words else None
    count_vect = CountVectorizer(stop_words=en_stopwords, binary=binary, token_pattern=r'\b\w[\']?\w*\b', min_df=min_df, max_df=max_df)
    dtm_train = count_vect.fit_transform(train_text)
    dtm_test = count_vect.transform(test_text)

    word_to_idx = count_vect.vocabulary_
    print("num of words:", len(word_to_idx))
    return dtm_train, dtm_test, word_to_idx, count_vect


def tfidf_vectorizer(train_text, test_text, stop_words=False, binary=False, min_df=3, max_df=0.95):
    en_stopwords = stopwords.words('english') if stop_words else None
    # sublinear_tf=True,
    tfidf_vect = TfidfVectorizer(stop_words=en_stopwords, binary=binary, token_pattern=r'\b\w[\']?\w*\b', norm='l2', min_df=min_df, max_df=max_df)
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
