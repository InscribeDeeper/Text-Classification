import mailparser
import re
from nltk.tokenize import word_tokenize
from nltk import pos_tag

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
    # x = " ".join(word_tokenize(x, preserve_line=True)).strip()
    x = " ".join(re.findall(r"(?u)\b\w+'[vnt]\w*\b|\b\w\w+\b[\d\.]+|\S+", x)).strip()  # this is the pattern that match shouldn't they're
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


#### Optional POS tag parser
def pos_tag_parser(text, target_tag):
    tokens = word_tokenize(text)
    # tokens = re.findall(r'\b\w[\']?\w*\b', text)
    tagged_tokens = pos_tag(tokens)
    return " ".join([word for word, tag in tagged_tokens if tag in target_tag])

