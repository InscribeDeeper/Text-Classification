import pandas as pd

from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS


def load_mystopwords(including=['the', 'ax']):
    """update stopwords
    """
    updated_sw = STOPWORDS.union(set(including))
    # updated_sw = set(stopwords.words('english') + including)
    return updated_sw


def upsampling_train(train, seeds=10):
    """upsampling for group with smaller size than the average group size

    Args:
        train ([pd.DataFrame]): [data corpus]
        seeds (int, optional): [random state]. Defaults to 10.

    Returns:
        [type]: [updated training set and the updated information]
    """
    group_size = train.groupby('label').size()
    mean_size = int(group_size.mean())
    small_groups = group_size[group_size < mean_size].index.tolist()

    train_small_groups = train[train['label'].isin(small_groups)].groupby('label').sample(n=mean_size, replace=True, random_state=seeds)
    train_large_groups = train[~train['label'].isin(small_groups)]
    upsampling_train = pd.concat([train_small_groups, train_large_groups], axis=0)
    upsampling_group_size = upsampling_train.groupby('label').size()
    upsampling_info = pd.concat([group_size, upsampling_group_size, upsampling_group_size - group_size], axis=1)
    upsampling_info.columns = ['before_upsampling', 'after_upsampling', 'increase']

    train = upsampling_train
    return train, upsampling_info


####################################
### columns selection
####################################

def load_data(only_stem_voc=False, train_path='../data/structured_train.json', test_path='../data/structured_test.json', sample50=False, \
                 select_cols=["global_index", "doc_path", "label", "reply", "reference_one", "reference_two", "tag_reply", "tag_reference_one", "tag_reference_two", "Subject", "From", "Lines", "Organization", "contained_emails", "long_string", "text", "error_message"]):
    """Load the processed dataset from datafolder
    
    Args:
        if only_stem_voc is True, we will load the processed dataset which are lemmatized, stemmed, and should be the existed vocbulary in wordnet

    """
    train = pd.read_json(train_path)
    test = pd.read_json(test_path)

    train['contained_emails'] = train['contained_emails'].apply(lambda x: " ".join(x) if x is not None else '')
    test['contained_emails'] = test['contained_emails'].apply(lambda x: " ".join(x) if x is not None else '')
    if sample50:
        seeds = 2021
        train = train.groupby('label').sample(50, random_state=seeds)
        test = test.groupby('label').sample(50, random_state=seeds)

    print("\nmay use cols: \n", select_cols)
    train = train[select_cols]
    train[["reply", "reference_one", "reference_two", "tag_reply", "tag_reference_one", "tag_reference_two", "Subject", "From", "Lines", "Organization"]] = train[["reply", "reference_one", "reference_two", "tag_reply", "tag_reference_one", "tag_reference_two", "Subject", "From", "Lines", "Organization"]].astype(str).fillna('')

    test = test[select_cols]
    test[["reply", "reference_one", "reference_two", "tag_reply", "tag_reference_one", "tag_reference_two", "Subject", "From", "Lines", "Organization"]] = test[["reply", "reference_one", "reference_two", "tag_reply", "tag_reference_one", "tag_reference_two", "Subject", "From", "Lines", "Organization"]].astype(str).fillna('')

    if only_stem_voc:
        stem_train, stem_test = load_stem_voc()
        train['text'] = stem_train
        test['text'] = stem_test

    return train, test


####################################
### data augmentation
####################################
def train_augmentation(train, select_comb=[['text'], ['reply', 'reference_one']]):
    """Based on combination of columns, this function will double the rows of samples

    Args:
        train ([type]): [description]
        select_comb (list, optional): [list of list, the inside list will represent which columns taht we need to combine]. Defaults to [['text'], ['reply', 'reference_one']].

    """
    if select_comb is None:
        return train['text'], train['label']

    dt = train.copy()
    train_text = pd.DataFrame()
    train_label = pd.DataFrame()
    for idx, comb in enumerate(select_comb):
        print(f"combination {idx+1} train: ", comb)
        dt = train[comb[0]]
        for col in comb[1:]:
            dt = dt + ' ' + train[col]
        train_text = pd.concat([train_text, dt], axis=0)
        train_label = pd.concat([train_label, train['label']], axis=0)

    train_text = train_text.reset_index(drop=True)[0]
    train_label = train_label.reset_index(drop=True)[0]
    return train_text, train_label


def load_stem_voc():
    stem_test = pd.read_csv("stem_voc_test.csv")['text'].apply(lambda x: " ".join(eval(x)))
    stem_train = pd.read_csv("stem_voc_train.csv")['text'].apply(lambda x: " ".join(eval(x)))
    return stem_train, stem_test
