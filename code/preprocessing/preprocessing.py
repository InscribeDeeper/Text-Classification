#!/usr/bin/env python
# coding: utf-8

# ## Package import
from glob import glob
import pandas as pd
from tqdm import tqdm
from parser_utils import typo_parser, email_address_parser,\
    bytedata_parser, structure_parser, pos_tag_parser, reference_parser


# ### main structural_email
def structural_email(data, pos_parser=True, bytedata_parser_threshold=50, reference_parser_match_type=2):
    """
    This is a parser pipeline, parser order matters.
    1. string => structure email to separate => header, body, others
    2. body => remove typo and some irrelevant words => body
    3. body => parse and remove email from body => body_no_email
    4. body_no_email => parse and remove binary data like BMP or picture from body => body_no_binary_no_email
    5. body_no_binary_no_email => separate email reference and reply => reply, previous_one, previous_two
    
    @param data: data text series including all the training set or test set
    @return: structural information
    """
    print("Preprocessing for unstructure email...")
    header_info = []
    body_info = []
    others_info = []
    tag_info = []
    for string in tqdm(data):
        # structure parsers
        header, body, others = structure_parser(string)
        body = typo_parser(body)
        body_no_email, emails = email_address_parser(body)
        body_no_binary_no_email, bytedata = bytedata_parser(body_no_email, threshold=bytedata_parser_threshold)

        # main parser
        reply, previous_one, previous_two = reference_parser(body_no_binary_no_email, match_type=reference_parser_match_type)
        if pos_parser:
            target_tag = set(['NN', 'NNS', 'NNPS'])
            tag_reply = pos_tag_parser(reply, target_tag)
            tag_previous_one = pos_tag_parser(previous_one, target_tag)
            tag_previous_two = pos_tag_parser(previous_two, target_tag)
            tag_info.append([tag_reply, tag_previous_one, tag_previous_two])

        # append data in loops
        header_info.append(header)
        body_info.append([reply, previous_one, previous_two])
        others_info.append(others + [emails] + [bytedata])

    a1 = pd.DataFrame.from_dict(header_info)
    a2 = pd.DataFrame(body_info, columns=["reply", "reference_one", "reference_two"])
    a3 = pd.DataFrame(others_info, columns=["date", "delivered_to", "to_domains", "error_message", "contained_emails", "long_string"])

    if pos_parser:
        a4 = pd.DataFrame(tag_info, columns=["tag_reply", "tag_reference_one", "tag_reference_two"])
        structure_email = pd.concat([a1, a2, a3, a4], axis=1)
    else:
        structure_email = pd.concat([a1, a2, a3], axis=1)

    return structure_email


# ## Data Loading
# - from file into DataFrame
def load_data_folder(path):
    """
    @param folders: the train or test directory
    @return: document list with [doc_path, doc, label, original_idx]
    """
    folders = glob(path + "/**")  # explore all the folder under the directory

    docs = []
    for classes in folders:
        label = classes.split("\\")[-1]
        doc_paths = glob(classes + "\\**")

        for doc_path in doc_paths:
            original_idx = doc_path.split("\\")[-1]

            with open(doc_path, encoding="UTF-8") as f:
                text = f.read()
            docs.append([doc_path, text, label, original_idx])

    print(f"\nLoaded folder under {path}: \n")
    for folder in folders:
        print(folder)

    return docs


corpus_train_docs = load_data_folder(path="../../data/train")
corpus_test_docs = load_data_folder(path="../../data/test")

# ## Preprocessing
corpus_train = pd.DataFrame(corpus_train_docs, columns=["doc_path", "text", "label", "original_idx"])
corpus_train = corpus_train.reset_index().rename(columns={"index": "global_index"})

corpus_test = pd.DataFrame(corpus_test_docs, columns=["doc_path", "text", "label", "original_idx"])
corpus_test = corpus_test.reset_index().rename(columns={"index": "global_index"})

print("original_idx duplicate count:", corpus_train.shape[0] - corpus_train.original_idx.drop_duplicates().shape[0], " on ", corpus_train.shape[0])
print("original_idx duplicate count:", corpus_test.shape[0] - corpus_test.original_idx.drop_duplicates().shape[0], " on ", corpus_test.shape[0])

# ### parsing
structural_train = structural_email(corpus_train["text"], pos_parser=True)
structural_test = structural_email(corpus_test["text"], pos_parser=True)

train = pd.concat([corpus_train, structural_train], axis=1)
test = pd.concat([corpus_test, structural_test], axis=1)
all_cols = train.columns.tolist()
print(all_cols)

# ## Saved processed data

train.to_json('../../data/structured_train.json')
test.to_json('../../data/structured_test.json')


# ## module test
def checking_text(idx, write_in_local=True):
    x = train[train["global_index"] == idx]
    string = x["text"].iloc[0]
    body = x["reply"].iloc[0]
    x_path = x["doc_path"].iloc[0]
    x_label = x["label"].iloc[0]

    if write_in_local:
        with open("./module_checking_sample.txt", "w", encoding="utf-8") as f:
            f.write(x_label + "\n\n")
            f.write(x_path + "\n\n")
            f.write(string)
    return string, body, x_path, x_label


module_test = True

if module_test:
    # 可以分开一个pyfile, 并且把这里的过程保存下来, 然后写在report中
    # idx = 22
    idx = 9187

    string, reply, x_path, x_label = checking_text(idx)

    header, body, others = structure_parser(string)
    print("\nrepr(header):   \n", repr(header))
    print("\nrepr(body):   \n", repr(body))
    print("\nrepr(others):   \n", repr(others))

    body = typo_parser(body)
    print("\nrepr(body):   \n", repr(body))

    body_no_email, emails = email_address_parser(body)
    print("\nrepr(body):   \n", repr(body))
    print("\nrepr(emails):   \n", repr(emails))
    print("\nrepr(body_no_email):   \n", repr(body_no_email))

    body_no_binary_no_email, bytedata = bytedata_parser(body_no_email, threshold=25)
    print("\nrepr(bytedata):   \n", repr(bytedata))
    print("\nrepr(body_no_binary_no_email):   \n", repr(body_no_binary_no_email))

    reply, previous_one, previous_two = reference_parser(body_no_binary_no_email, match_type=2)

    print("\nrepr(reply):   \n", repr(reply))
    print("\nrepr(previous_one):   \n", repr(previous_one))
    print("\nrepr(previous_two):   \n", repr(previous_two))

with open('regex_sample.txt', 'r') as f:
    sample = f.read()

parsed_f = structural_email(pd.Series(sample))
parsed_f.to_json("regex_sample_parsed.json")
