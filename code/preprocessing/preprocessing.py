#!/usr/bin/env python
# coding: utf-8

# ## Package import

import pandas as pd
from glob import glob
from tqdm import tqdm
import mailparser
import re
from nltk.tokenize import word_tokenize

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

# ### parsing

corpus_train = pd.DataFrame(corpus_train_docs, columns=["doc_path", "text", "label", "original_idx"])
corpus_train = corpus_train.reset_index().rename(columns={"index": "global_index"})

corpus_test = pd.DataFrame(corpus_test_docs, columns=["doc_path", "text", "label", "original_idx"])
corpus_test = corpus_test.reset_index().rename(columns={"index": "global_index"})

print("original_idx duplicate count:", corpus_train.shape[0] - corpus_train.original_idx.drop_duplicates().shape[0], " on ", corpus_train.shape[0])
print("original_idx duplicate count:", corpus_test.shape[0] - corpus_test.original_idx.drop_duplicates().shape[0], " on ", corpus_test.shape[0])

# #### typo_parser


def typo_parser(x):
    """
    1. replace irrelevant symbol "|" or "*"
    2. remove extra space "  "
    3. replace extra \n "\n\n" into "\n"
    4. replace "> *>" into ">>" for further analysis

    @param string: email body string
    @return: cleaned email body string, extracted emails
    
    # test_string = 'www.\n com\n\n or ?\n>\n    >>\n    \n > > >|> (note) \n> \n I\nam not good enough with regex>'
    # typo_parser(test_string)

    """
    # x = re.sub('([,:;?!\.”\)])\n', '\g<1> ', x)  # add space for symbol like .\n or ?\n
    # x = re.sub('(\w)\n(\w)', '\g<1> \g<2>', x)  # add space for symbol like word\nword
    x = re.sub('\n', ' \n ', x)  # add space for between \n
    x = re.sub("[\*|\|\^]", "", x)  # replace irrelevant symbol "|" or "*"

    x = re.sub(">[ >]*>", ">>", x)  # compress > [?] >
    x = re.sub("\[.*?\]", "", x, flags=re.S)  # separate for typo like [a)
    x = re.sub("\(.*?\)", "", x, flags=re.S)

    x = re.sub("\n[ \n]*\n", "\n", x)  # compress \n
    return x


# #### email_address_parser


def email_address_parser(string):
    """
    extract and remove email from the body
    @param string: email body string
    @return: cleaned email body string, extracted emails
    """
    emails = None
    emails = re.findall(" ?[\S]+@[\S]+ ?", string)
    string = re.sub(" ?[\S]+@[\S]+ ?", " ", string)
    return string, emails


# #### bytedata_parser


def bytedata_parser(string, threshold=50):
    """
    Since 99% of english words length ranged from [1,20], but consider special symbol there, we set the threshold with 50 for only parse bytdata like photo
    If length of span larger than threshold, then we will not treat it as a word. 
    sep can only use space
    """
    bytedata = None
    clean_string = " ".join([word for word in re.split(" ", string) if len(word) <= threshold])
    ## sentence length is the same
    # clean_string = "\n".join([word for word in re.split("\n", clean_string) if len(word)<=threshold])
    bytedata = [word for word in re.split(" ", string) if len(word) > threshold]
    return clean_string, bytedata


# #### structure_parser


def structure_parser(string):
    """
    @param parser: email string
    @return: structural information for email header, body, others
    """
    error_message = None
    header = {}
    body = ""
    others = []
    try:
        mail = mailparser.parse_from_string(string)
        if mail.has_defects:  # [first line error]
            remove_first_line_string = "\n".join(string.split("\n")[1:])
            mail = mailparser.parse_from_string(remove_first_line_string)
            # print("remove_first_line_string update for ")
        header, body = mail.headers, mail.body
        others = [mail.date, mail.delivered_to, mail.to_domains, error_message]

    except Exception as error:
        error_message = error
    return header, body, others


# #### reference_parser


def tokenizer_parser(x):
    """
    remove_flag e.g. In article
    remove extra space in the middle 
    remove special symbol
    """
    x = re.sub("(?:In article)?.*writes:", "", x, flags=re.S)
    # x = re.sub(" {2,}", " ", x) # compress space
    x = " ".join(word_tokenize(x, preserve_line=True)).strip()
    return x


def reference_parser(string, match_type=2):
    """
    Consider reply with referencing previous email, we need to separate them to make prediction separately.
    @param 
        string: email body string
        match_type: 0 with return only main body, 1 with return main body + previous one reference, 2 with more reference
    @return: 
        reply, previous_one, previous_two in the email
    
    @ test with the following code
    string = " \n\n\n\n    >>>zero email \n\n >>first email\n >second email\n reply email \n"
    reply, previous_one, previous_two = reference_parser(string, match_type=2)
    print("## reply\n", repr(reply))
    print("## previous_one\n", repr(previous_one))
    print("## previous_two\n", repr(previous_two))
    """

    previous_one, previous_two, reply = '', '', ''

    # extract reply with out containing >
    reply = " ".join([s for s in string.split("\n") if ">" not in s])
    reply = tokenizer_parser(reply)

    # add "\n" before string to matchign [^>]{1}
    if match_type > 0:
        previous_one = " ".join(re.findall("[^>]{1}>{1}([^>]{1}[\S ]*)\n", "\n" + string))  # matching >
        previous_one = tokenizer_parser(previous_one)

    if match_type > 1:  # flag reference_two
        previous_two = " ".join(re.findall("[^>]{1}>{2}([^>]{1}[\S ]*)\n", "\n" + string))  # matching >>
        previous_two = tokenizer_parser(previous_two)
    # previous_two_more_pt = "[^>]{1}>{2,}([^>]{1}[\S ]*)\n" # matching >> or >>> more
    return reply, previous_one, previous_two


# ### main structural_email
def structural_email(data, bytedata_parser_threshold=50, reference_parser_match_type=2):
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
    for string in tqdm(data):
        # structure parsers
        header, body, others = structure_parser(string)
        body = typo_parser(body)
        body_no_email, emails = email_address_parser(body)
        body_no_binary_no_email, bytedata = bytedata_parser(body_no_email, threshold=bytedata_parser_threshold)

        # main parser
        reply, previous_one, previous_two = reference_parser(body_no_binary_no_email, match_type=reference_parser_match_type)

        # append data in loops
        header_info.append(header)
        body_info.append([reply, previous_one, previous_two])
        others_info.append(others + [emails] + [bytedata])

    a1 = pd.DataFrame.from_dict(header_info)
    a2 = pd.DataFrame(body_info, columns=["reply", "reference_one", "reference_two"])
    a3 = pd.DataFrame(others_info, columns=["date", "delivered_to", "to_domains", "error_message", "contained_emails", "long_string"])
    structure_email = pd.concat([a1, a2, a3], axis=1)
    return structure_email


# ## Main block

structural_train = structural_email(corpus_train["text"])
structural_test = structural_email(corpus_test["text"])

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
