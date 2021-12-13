# encoding=utf-8
import random
from tensorflow.keras.utils import model_to_dot
from IPython.display import SVG
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Input, Flatten, Concatenate
from tensorflow.keras.models import Model
import gensim.downloader as api
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec


##################################################
# Dataset prepare class
##################################################
class text_preprocessor(object):
    def __init__(self, doc_len, max_words, docs, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', char_level=False, zero_pad=None):
        '''

        @param 
            doc_len, max_words: The doc_len and max_words can be obtain from eda_utils.py if we need to cut by percentage
            docs: The training corpurs. Data Series prefer
        @return: 
            processor instance initialized by the training corpus. 
        @method:            
            generate_seq: convert new docs into sequence with the same length as the setting
            w2v_pretrain: train by corpus by gensim CBOW or SKIPGRAM model and return the pretrain embedding matrix
            load_glove_w2v: load glove embedding

        '''

        self.MAX_DOC_LEN = doc_len
        self.MAX_NB_WORDS = max_words
        self.tokenizer = Tokenizer(num_words=self.MAX_NB_WORDS, filters=filters, char_level=char_level)
        self.tokenizer.fit_on_texts(docs)
        self.corpus = docs

        if zero_pad:  # for cnn this is not needed
            self.tokenizer.word_index[zero_pad] = 0
            self.tokenizer.index_word[0] = zero_pad

        self.index_word = self.tokenizer.index_word
        self.word_index = self.tokenizer.word_index

    def __repr__(self):
        return 'A class which has method:\n' \
               'generate_seq(sentences_train) \n' \
               'w2v_pretrain(dimension of embedding)\n' \
               'load_glove_w2v(dimension of embedding)\n' \
               ''

    def generate_seq(self, docs, padding='post', truncating='post'):
        sequences = self.tokenizer.texts_to_sequences(docs)
        # padded_sequences = pad_sequences(sequences, maxlen=self.MAX_DOC_LEN, padding='post')
        padded_sequences = pad_sequences(sequences, maxlen=self.MAX_DOC_LEN, padding=padding, truncating=truncating)
        return padded_sequences

    def w2v_pretrain(self, EMBEDDING_DIM, min_count=1, seed=1, cbow_mean=1, negative=5, window=5, workers=8):
        # Generate pretrained Embedding with all of tokens in training sentences
        wv_model = word2vec.Word2Vec(sentences=self.corpus, min_count=min_count, seed=seed, cbow_mean=cbow_mean, size=EMBEDDING_DIM, negative=negative, window=window, workers=workers)  # Based on tokens in all sentences, training the W2V # sg = 1 为 skipgram
        NUM_WORDS = min(self.MAX_NB_WORDS, len(self.tokenizer.word_index))  # Keep the # of highest freq of words
        embedding_matrix = np.zeros((NUM_WORDS + 1, EMBEDDING_DIM))  # "+1" is for padding symbol that equal 0
        # embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))  #  RNN not needed

        for word, i in self.tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            if word in wv_model.wv:
                embedding_matrix[i] = wv_model.wv[word]  # load pretrained embedding on my indexing table embedding
        return embedding_matrix

    def load_glove_w2v(self, EMBEDDING_DIM):
        word_vectors = api.load("glove-wiki-gigaword-" + str(EMBEDDING_DIM))  # load pre-trained word-vectors from glove
        NUM_WORDS = min(self.MAX_NB_WORDS, len(self.tokenizer.word_index))  # Keep the # of highest freq of words
        embedding_matrix = np.zeros((NUM_WORDS + 1, EMBEDDING_DIM))  # "+1" is for padding symbol that equal 0
        # embedding_matrix = np.zeros((NUM_WORDS, EMBEDDING_DIM))  #  RNN not needed

        for word, i in self.tokenizer.word_index.items():
            if i >= NUM_WORDS:
                continue
            try:
                embedding_matrix[i] = word_vectors.get_vector(word)
            except Exception:
                embedding_matrix[i] = np.random.random(EMBEDDING_DIM)
        return embedding_matrix


##################################################
# define CNN part
##################################################


def cnn_model(
    FILTER_SIZES,
    MAX_NB_WORDS,
    MAX_DOC_LEN,
    NAME='Multi_chanel_textCNN',
    EMBEDDING_DIM=200,
    NUM_FILTERS=64,
    PRETRAINED_WORD_VECTOR=None,
    trainable_switch=True,
):
    '''
        @param 
            FILTER_SIZES: kernel size for each convolution filter
            MAX_NB_WORDS: the vocabulary size
            MAX_DOC_LEN: max length of each sample, the words larger than this length will be truncated
            EMBEDDING_DIM: embedding shape
            NUM_FILTERS: how many different filters in the convolution layer for oen kernel size
            PRETRAINED_WORD_VECTOR: init the embedding layer by pretrain embedding matrix
            trainable_switch: fix embedding layer or not
        @return: 
            the Multi_chanel_textCNN module as feature extractor. 
            input shape is the (MAX_DOC_LEN, )
            output shape is the the number of different shape of filters * number of filters for each shape. e.g. filter size = [2,3,4], num_filters = 64, then the output shape will be 3*64
    '''
    main_input = Input(shape=(MAX_DOC_LEN, ), name='main_input')

    if (PRETRAINED_WORD_VECTOR is not None):
        embed_1 = Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, mask_zero=True, input_length=MAX_DOC_LEN, name=f'pretrained_embedding_trainable_{trainable_switch}', weights=[PRETRAINED_WORD_VECTOR], trainable=trainable_switch)(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, embeddings_initializer='uniform', mask_zero=True, input_length=MAX_DOC_LEN, name='embedding_trainable', trainable=True)(main_input)

    conv_blocks = []
    for f in FILTER_SIZES:  # For every filter
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, name='conv_' + str(f) + '_gram', strides=1, activation='relu')(embed_1)  # convolution  # filter-kernal extracting 64 features with ReLU activation function
        pool = MaxPooling1D(pool_size=MAX_DOC_LEN - f + 1, name='pool_' + str(f) + '_gram')(conv)  # maxpooling size = MAX_DOC_LEN - filter_size + 1
        flat = Flatten(name='flat_' + str(f) + '_gram')(pool)  # flatten filters extracting features (size*number = 3*64)
        conv_blocks.append(flat)

    if len(conv_blocks) > 1:
        z = Concatenate(name='concate')(conv_blocks)  # Concatenate list of feature maps [flat_1, flat_2, flat_3]
    else:
        z = conv_blocks[0]

    # pred = Dense(20, activation='softmax')(z)
    model = Model(inputs=main_input, outputs=z, name=NAME)

    return model


# testing
# cnn_base = cnn_model(FILTER_SIZES=[2,3,4], NUM_FILTERS=64, MAX_DOC_LEN=MAX_DOC_LEN, MAX_NB_WORDS=MAX_NB_WORDS, EMBEDDING_DIM=300)
# cnn_base.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# cnn_base.fit(x=X_test,y=y_test,batch_size=20)
# cnn_base.summary()
# plot_model(cnn_base,show_shapes=True)


##################################################
# training history plot from keras
##################################################
def history_plot(training, extra_metric=None):
    '''
        @param 
            FILTER_SIZES: kernel size for each convolution filter
            MAX_NB_WORDS: the vocabulary size
            MAX_DOC_LEN: max length of each sample, the words larger than this length will be truncated
            EMBEDDING_DIM: embedding shape
            NUM_FILTERS: how many different filters in the convolution layer for oen kernel size
            PRETRAINED_WORD_VECTOR: init the embedding layer by pretrain embedding matrix
            trainable_switch: fix embedding layer or not
        @return: 
            processor instance initialized by the training corpus. 
        @method:            
            generate_seq: convert new docs into sequence with the same length as the setting
            w2v_pretrain: train by corpus by gensim CBOW or SKIPGRAM model and return the pretrain embedding matrix
            load_glove_w2v: load glove embedding

    # ################# plot training history
    #     dic = ['val_loss', 'loss', 'val_acc', 'acc', "val_auroc"] # print(training.history)
    #     loss: 0.8109 - acc: 0.6362 - auroc: 0.7960 - val_loss: 0.6793 - val_acc: 0.7144 - val_auroc: 0.8684
    '''
    dic = list(training.history.keys())

    if extra_metric is not None:
        idx = [[0, 3], [1, 4], [2, 5]]
    else:
        idx = [[0, 2], [1, 3]]

    for i, j in idx:
        print("========================================================================")
        print(dic[i], dic[j])
        xx = list(range(1, len(training.history[dic[i]]) + 1))
        plt.plot(xx, training.history[dic[i]], color='navy', lw=2, label='Model_' + str(dic[i]))
        plt.plot(xx, training.history[dic[j]], color='darkorange', lw=2, label='Model_' + str(dic[j]))
        plt.title(str(dic[i]) + "v.s. training_" + str(dic[j]))
        plt.xlabel('Epochs')
        plt.ylabel(str(dic[i]))
        plt.legend()
        plt.show()
    return None


def cnn_model_l2(
    FILTER_SIZES,
    MAX_NB_WORDS,
    MAX_DOC_LEN,
    NAME='Multi_chanel_textCNN',
    EMBEDDING_DIM=200,
    NUM_FILTERS=64,
    PRETRAINED_WORD_VECTOR=None,
    trainable_switch=True,
):
    '''
        @param 
            FILTER_SIZES: kernel size for each convolution filter
            MAX_NB_WORDS: the vocabulary size
            MAX_DOC_LEN: max length of each sample, the words larger than this length will be truncated
            EMBEDDING_DIM: embedding shape
            NUM_FILTERS: how many different filters in the convolution layer for oen kernel size
            PRETRAINED_WORD_VECTOR: init the embedding layer by pretrain embedding matrix
            trainable_switch: fix embedding layer or not
        @return: 
            the Multi_chanel_textCNN module as feature extractor. 
            input shape is the (MAX_DOC_LEN, )
            output shape is the the number of different shape of filters * number of filters for each shape. e.g. filter size = [2,3,4], num_filters = 64, then the output shape will be 3*64
    '''
    main_input = Input(shape=(MAX_DOC_LEN, ), name='main_input')

    if (PRETRAINED_WORD_VECTOR is not None):
        embed_1 = Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, mask_zero=True, input_length=MAX_DOC_LEN, name=f'pretrained_embedding_trainable_{trainable_switch}', weights=[PRETRAINED_WORD_VECTOR], trainable=trainable_switch)(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS, output_dim=EMBEDDING_DIM, embeddings_initializer='uniform', mask_zero=True, input_length=MAX_DOC_LEN, name='embedding_trainable', trainable=True)(main_input)

    conv_blocks = []
    for f in FILTER_SIZES:  # For every filter
        x = Conv1D(filters=NUM_FILTERS, kernel_size=f, name='conv_' + str(f) + '_gram1', strides=1, activation='relu')(embed_1)  # convolution  # filter-kernal extracting 64 features with ReLU activation function
        x = MaxPooling1D(pool_size=f * 3, name='pool_' + str(f) + '_gram1')(x)  # maxpooling size = MAX_DOC_LEN - filter_size + 1
        x = Conv1D(filters=NUM_FILTERS * 2, kernel_size=f, name='conv_' + str(f) + '_gram2', strides=2, activation='relu')(x)  # convolution  # filter-kernal extracting 64 features with ReLU activation function
        x = MaxPooling1D(pool_size=f * 2, name='pool_' + str(f) + '_gram2')(x)  # maxpooling size = MAX_DOC_LEN - filter_size + 1
        flat = Flatten(name='flat_' + str(f) + '_gram')(x)  # flatten filters extracting features (size*number = 3*64)
        conv_blocks.append(flat)

    if len(conv_blocks) > 1:
        z = Concatenate(name='concate')(conv_blocks)  # Concatenate list of feature maps [flat_1, flat_2, flat_3]
    else:
        z = conv_blocks[0]

    # pred = Dense(20, activation='softmax')(z)
    model = Model(inputs=main_input, outputs=z, name=NAME)

    return model


def visual_textCNN(model, filename='multichannel-CNN.png'):
    print(model.summary())
    return SVG(model_to_dot(model, dpi=70, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg'))


def setup_seed_ml(seed):
    np.random.seed(seed)
    random.seed(seed)
    return
