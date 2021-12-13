# nltk.download('stopwords')

from data_utils import load_mystopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt
from gensim.parsing.preprocessing import remove_stopwords


from nltk import word_tokenize

# def remove_stopwords(sent, stopwords):
#     sent = " ".join([word for word in word_tokenize(sent, preserve_line=True) if word not in stopwords]) # 这个tokenizer做了很多不行的东西
#     return sent


def eda_MAX_NB_WORDS(corpus, remove_stop=False, ratio=0.95, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=False):
    '''
    Input: data series
    '''
    #
    if remove_stop:
        en_stopwords = load_mystopwords()
        corpus = corpus.apply(lambda x: remove_stopwords(x))

    tokenizer_eda = Tokenizer(num_words=None, filters=filters, lower=True, char_level=char_level)  # 如果有这个, NLTK的preprocessing可以不用做
    tokenizer_eda.fit_on_texts(corpus)
    b = pd.DataFrame(tokenizer_eda.word_counts.items(), columns=['word', 'count'])

    # by nltk
    #     tfidf_vect = TfidfVectorizer(stop_words="english", norm=None, min_df=0, max_df=0.999, use_idf=False, smooth_idf=False)
    #     dtm = tfidf_vect.fit_transform(corpus)
    #     b = pd.DataFrame.from_dict(tfidf_vect.vocabulary_, orient='index').reset_index()
    #     b.columns =['word', 'count']
    # grouping

    # 排序重建index 就是 tokenizer中的word_index
    a = b.sort_values(by='count', ascending=False).reset_index()

    # ########### 累加百分比 可视化
    plt.figure(figsize=(20, 5))
    word_distribution = a['count'].cumsum() / a['count'].sum()  # 求累加百分比
    word_distribution.plot()  # 出图

    # cut_index = np.argmin(abs(word_distribution-ratio)) # 找到离0.8最近的index位置
    diff = abs(word_distribution - ratio)
    cut_index = diff[diff == min(diff)].index[0]

    plt.plot([cut_index, cut_index], [0, ratio])  # 找出 固定 ratio 的index
    plt.plot([0, cut_index], [ratio, ratio])
    plt.xlabel("word_index")  # 需要先sort, 才能说是index of words.
    plt.ylabel("word_cum_counts_perc")
    plt.title("MAX_NB_WORDS Cumsum Percentage")
    plt.show()

    # ############  大概取词范围 可视化
    plt.figure(figsize=(20, 5))
    b.iloc[:, 1].plot()  # 出图
    # 找出 固定 ratio 的index
    plt.plot([cut_index, cut_index], [0, max(b['count'])])
    plt.plot([0, cut_index], [max(b['count']), max(b['count'])])
    plt.xlabel("word_index")  # 需要先sort, 才能说是index of words.
    plt.ylabel("word_count")
    plt.title("MAX_NB_WORDS Percentage")
    plt.show()
    print("Cut index with", ratio * 100, "% of corpus: ", cut_index, '\n')
    # stopwords?
    print(a.sort_values(by='count', ascending=False).head(20))
    print("extreme frequent words:", b[b['count'] > 50000])
    return int(cut_index)
    # return int(cut_index)+1


def eda_MAX_DOC_LEN(corpus, remove_stop=False, ratio=0.9, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=False):
    '''
    Input: list of sentences (string type)
    '''

    if remove_stop:
        en_stopwords = load_mystopwords()
        corpus = corpus.apply(lambda x: remove_stopwords(x))

    # by keras
    tokenizer_eda = Tokenizer(num_words=None, filters=filters, lower=True, char_level=char_level)
    tokenizer_eda.fit_on_texts(corpus)
    dt_q1 = pd.DataFrame([len(i) for i in tokenizer_eda.texts_to_sequences(corpus)], columns=['length'])

    # grouping
    c = dt_q1['length'].value_counts().sort_index()  # 频数统计, 且按index重新排序
    sent_cdf = c.cumsum() / c.sum()
    sent_pdf = c / c.sum()
    # cut_index = np.argmin(abs(sent_cdf - ratio))  # 找到离0.8最近的index位置
    diff = abs(sent_cdf - ratio)
    cut_index = diff[diff == min(diff)].index[0]

    # ########### 累加百分比 可视化
    plt.figure(figsize=(20, 5))
    sent_cdf.plot()  # 出图

    plt.plot([cut_index, cut_index], [0, ratio])  # 找出 固定 ratio 的index
    plt.plot([0, cut_index], [ratio, ratio])
    plt.xlabel("DOC_length")  # 需要先sort, 才能说是index of words.
    plt.ylabel("DOC_cumsum_length_perc")
    plt.title("MAX_DOC_LEN CDF")
    plt.show()

    plt.figure(figsize=(20, 5))
    sent_pdf.plot()  # 出图
    plt.plot([cut_index, cut_index], [0, max(sent_pdf)])  # 找出 固定 ratio 的index
    plt.plot([0, cut_index], [max(sent_pdf), max(sent_pdf)])  # 横线
    plt.xlabel("DOC_length")  # 需要先sort, 才能说是index of words.
    plt.ylabel("DOC_length_perc")
    plt.title("MAX_DOC_LEN PDF")
    plt.show()

    print("Cut index with", ratio * 100, "% of corpus: ", cut_index)
    return int(cut_index)


def eda_WORD_CLOUD(corpus, remove_stop=False, color='Purples', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):

    en_stopwords = None

    if remove_stop:
        en_stopwords = load_mystopwords()
        corpus = corpus.apply(lambda x: remove_stopwords(x))

    # 如果有这个, NLTK的preprocessing可以不用做    tokenizer_eda.fit_on_texts(corpus)
    tokenizer_eda = Tokenizer(num_words=None, filters=filters, lower=True)
    tokenizer_eda.fit_on_texts(corpus)
    b = pd.DataFrame(tokenizer_eda.word_counts.items(), columns=['word', 'count'])
    # 排序重建index 就是 tokenizer中的word_index
    a = b.sort_values(by='count', ascending=False).reset_index()
    sorted_freq = a
    sorted_freq.columns = ['idx', 'word', 'freq']
    top1000_freq = np.log(sorted_freq[['word', 'freq']].head(10000).set_index('word')).to_dict()['freq']

    #     wordcloud = WordCloud(scale=15, width=400, height=200, relative_scaling=1, prefer_horizontal=0.8, mask = None,
    #                       min_font_size=1, max_font_size=30, background_color='white', colormap='tab20_r');
    wordcloud = WordCloud(
        scale=30,
        width=400,
        height=200,
        relative_scaling=1,
        prefer_horizontal=0.8,
        mask=None,
        #                       min_font_size=1, max_font_size=30, stopwords=en_stopwords,background_color='white', colormap='purple');
        min_font_size=1,
        max_font_size=30,
        background_color='white',
        colormap=color)
        # colormap=color) # Purples / Blues
    wordcloud.fit_words(top1000_freq)  # 把数据输入 wordcloud 生成器
    plt.figure(figsize=(20, 10))
    # 图片大小
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    return None


# eda_MAX_NB_WORDS(corpus = filtered_corpus, ratio = 0.95)
# eda_MAX_DOC_LEN(corpus = filtered_corpus, ratio=0.9)
# eda_WORD_CLOUD(corpus)
