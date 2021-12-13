import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import remove_stopwords
from data_utils import load_mystopwords
####################################
### string normalized
####################################
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')


def normal_string(x):
    """remove stopwords and remove special tokens. like irrelevant symbol

    Args:
        x ([string]): [a sentence]

    Returns:
        [string]: [sentence after removing stopwords and special symbol filtered by nltk word tokenizer]
    """

    x = remove_stopwords(x)
    #     x = " ".join(preprocess_string(x))
    x = " ".join(word_tokenize(x, preserve_line=False)).strip()
    return x


def extract_stem_voc(x):
    """extract word from predefined vocbulary with stemming and lemmatization

    Args:
        x ([string]): [a sentence]

    Returns:
        [list]: [word after stemming and lemmatization]
    """
    stem = PorterStemmer()
    #     wnl = WordNetLemmatizer()
    all_words = set(words.words())
    #     lemma_word = [word for word in map(lambda x: wnl.lemmatize(stem.stem(x)), re.findall('[a-zA-Z][-._a-zA-Z]*[a-zA-Z]', x)) if word in all_words]
    lemma_word = [word for word in map(lambda x: stem.stem(x), re.findall('[a-zA-Z][-._a-zA-Z]*[a-zA-Z]', x)) if word in all_words]
    return lemma_word


def count_vectorizer(train_text, test_text, dense=False, voc=None, stop_words=False, binary=False, min_df=3, max_df=0.95, ngram_range=(1, 1)):
    """
    Sparse matrix is need for avoiding OOM 

    Args:
        train_text ([pd.series]): [original train text or select columns]
        test_text ([pd.series]): [original test text or select columns]
        dense (bool, optional): [output sparse dtm or not]. Defaults to False.
        voc ([dict], optional): [predefined vocabulary]. Defaults to None.
        stop_words (bool, optional): [remove stopwords or not]. Defaults to False.
        binary (bool, optional): [binarlized TF or not ]. Defaults to False.
        min_df (int, optional): [minimum frequency accross all document]. Defaults to 3.
        max_df (float, optional): [maximum frequency ratio accross all document]. Defaults to 0.95.
        ngram_range (tuple, optional): [n-gram ]. Defaults to (1, 1).

    Returns:
        [array, array, dict, class]: [tf matrix for train and test, the index mapping between words and tf columns idx, the fitted vectorizer class]
    """
    en_stopwords = load_mystopwords() if stop_words else None
    count_vect = CountVectorizer(stop_words=en_stopwords, binary=binary, vocabulary=voc, token_pattern=r'\b\w[\']?\w*\b', min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    dtm_train = count_vect.fit_transform(train_text)
    dtm_test = count_vect.transform(test_text) if test_text is not None else None

    word_to_idx = count_vect.vocabulary_
    print("num of words:", len(word_to_idx))
    if dense:
        dtm_train = dtm_train.toarray()
        dtm_test = dtm_test.toarray()
    return dtm_train, dtm_test, word_to_idx, count_vect


def tfidf_vectorizer(train_text, test_text, dense=False, voc=None, stop_words=False, binary=False, min_df=3, max_df=0.95, ngram_range=(1, 1)):
    """
    Sparse matrix is need for avoiding OOM 

    Args:
        train_text ([pd.series]): [original train text or select columns]
        test_text ([pd.series]): [original test text or select columns]
        dense (bool, optional): [output sparse dtm or not]. Defaults to False.
        voc ([dict], optional): [predefined vocabulary]. Defaults to None.
        stop_words (bool, optional): [remove stopwords or not]. Defaults to False.
        binary (bool, optional): [binarlized TF or not ]. Defaults to False.
        min_df (int, optional): [minimum frequency accross all document]. Defaults to 3.
        max_df (float, optional): [maximum frequency ratio accross all document]. Defaults to 0.95.
        ngram_range (tuple, optional): [n-gram ]. Defaults to (1, 1).

    Returns:
        [array, array, dict, class]: [tfidf matrix for train and test, the index mapping between words and tfidf columns idx, the fitted vectorizer class]
    """
    en_stopwords = load_mystopwords() if stop_words else None
    # sublinear_tf=True,
    tfidf_vect = TfidfVectorizer(stop_words=en_stopwords, binary=binary, vocabulary=voc, token_pattern=r'\b\w[\']?\w*\b', norm='l2', min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    dtm_train = tfidf_vect.fit_transform(train_text)
    dtm_test = tfidf_vect.transform(test_text) if test_text is not None else None

    word_to_idx = tfidf_vect.vocabulary_
    print("num of words:", len(word_to_idx))
    if dense:
        dtm_train = dtm_train.toarray()
        dtm_test = dtm_test.toarray()
    return dtm_train, dtm_test, word_to_idx, tfidf_vect


def dimension_reduction(dtm, method='svd', out_dim=200, verbose=0):
    """dimension reduction for matrix 

    Args:
        dtm ([np.array]): [tfidf or tf matrix]
        method (str, optional): [svd or tsne, but tsne only for visualization]. Defaults to 'svd'.
        out_dim (int, optional): [target dimension]. Defaults to 200.
        verbose (int, optional): [for more detail middle processing information ]. Defaults to 0.

    Returns:
        [np.array, class]: [dtm after dimension reduction, mapper ]
    """
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


def feature_selection_chi2(vectorizer, X_train, y_train, X_test, k=5000):
    """feature selection based on chi2
    """
    feature_names = vectorizer.get_feature_names()
    # Feature reduction with Kbest features based on chi2 score
    ch2 = SelectKBest(chi2, k=k)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    if feature_names:
        feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]
    return feature_names
