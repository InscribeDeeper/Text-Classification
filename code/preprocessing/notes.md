# Preprocessing

With regular expression and mailparser, the preprocessing mainly parse the structure of email and clean the text.

The core features are:

- 	Indentify the Structure of the email and extract specific part of email content
-   Identify previous email reference and separate as different features (up to 2)
- 	Clean extra symbol, and format the sentences by NLTK word tokenizer
-   Saved the preprocessed train and test file in local
-   We generated columns:
    ```python
    ['global_index', 'doc_path', 'text', 'label', 'original_idx', 'From', 'Subject', 'Summary', 'Keywords', 'Expires', 'Distribution', 'Organization', 'Supersedes', 'Lines', 'X-Newsreader', 'NNTP-Posting-Host', 'Reply-To', 'Nntp-Posting-Host', 'In-Reply-To', 'News-Software', 'X-Mailer', 'Originator', 'Article-I.D.', 'X-News-Reader', 'X-Sender', 'X-Disclaimer', 'Nntp-Posting-User', 'X-Bytes', 'X-Xxmessage-Id', 'X-Xxdate', 'X-Useragent', 'In-reply-to', 'OD-Comment-To', 'ReplyTo', 'Disclaimer', 'Comments', 'Posting-Front-End', 'X-Reader', 'Mime-Version', 'Content-Type', 'Content-Transfer-Encoding', 'X-UserAgent', 'X-NewsSoftware', 'Nntp-Software', 'Oganization', 'Apparently-To', 'X-Comment-To', 'X-Gateway', 'X-Advert', 'Cc', 'X-News-Software', 'X-Posted-From', 'Follow-Ups', 'X-Auth-User', 'X-FTN-To', 'X-Gated-By', 'X-Standard-Disclaimer', 'Moderator', 'X-XXMessage-ID', 'X-XXDate', 'To', 'Posted-Date', 'Received-Date', 'Orginization', 'X-Md4-Signature', 'Return-Receipt-To', 'X-Mail-Reader', 'Content-Length', 'X-Copyright', 'Original-To', 'X-Received', 'X-To', 'Return-Path', 'Nntp-Posting-Host-[nntpd-23809]', 'Organisation', 'X-Date', 'Nntp-Posting-Host-[nntpd-8755]', 'Nntp-Posting-Host-[nntpd-19510]', 'Nntp-Posting-Host-[nntpd-29970]', 'X-Software', 'X-AltNet-ID', 'MIME-Version', 'Bcc', 'Status', 'Nntp-Posting-Host-[nntpd-681]', 'Weather', 'Moon-Phase', 'X-Last-Updated', 'X-Face', 'X-Maildoor', 'X-Newssoftware', 'Nf-ID', 'Nf-From', 'X-Address', 'X-Fax', 'X-Phone', 'IMPORTANT-INFO', 'X-Added', 'Original-Sender', 'X-Alt.reply-Address', 'X-X-From', 'Mmdf-Warning', 'Followups-to', 'X-Newsposter', 'X-Header', 'X-Cc', 'Oanization', 'reply', 'reference_one', 'reference_two', 'date', 'delivered_to', 'to_domains', 'error_message', 'contained_emails', 'long_string']
    ```
-   Index are duplicated, so I created another global index for further data retrieve and checking
    -   original_idx duplicate count: 1060 on 11083
    -   original_idx duplicate count: 770 on 7761

Drawback:

-   The order of regex are important. In other words, they are dependent. It is not that easy to maintain.

Details(to do):

-   The connection between lines have no space.


---

# Sample

Sample Input:

``` txt
From: pyron@skndiv.dseg.ti.com (Dillon Pyron)
Subject: Re: Founding Father questions
Lines: 35
Nntp-Posting-Host: skndiv.dseg.ti.com
Reply-To: pyron@skndiv.dseg.ti.com
Organization: TI/DSEG VAX Support


In article <1993Apr5.153951.25005@eagle.lerc.nasa.gov>, pspod@bigbird.lerc.nasa.gov (Steve Podleski) writes:
>arc@cco.caltech.edu (Aaron Ray Clements) writes:
>>Wasn't she the one making the comment in '88 about George being born with
>>a silver foot in his mouth?  Sounds like another damn politician to me.
>>
>>Ain't like the old days in Texas anymore.  The politicians may have been
>>corrupt then, but at least they'd take a stand.  (My apologies to a few
>>exceptions I can think of.)  
>>
>>News now is that the House may already have a two-thirds majority, so 
>>her "opposition" out of her concern for image (she's even said this
>>publicly) may not matter.
>
>Do people expect the Texans congressmen to act as the N.J. Republicans did?

There is a (likely) veto proof majority in the house.  The Senate,
unfortunately, is a different story.  The Lt.Gov. has vowed that the bill will
not be voted on, and he has the power to do it.  In addition, the Senate is a
much smaller, and more readily manipulated body.

On ther other hand, the semi-automatic ban will likely not live, as at least
fifty per cent of the house currently opposes it, and it is VERY far down in
the bill order in the Senate (I believe it will be addressed after the CCW
bill).

And I thought my TX Political Science class was a waste of time!
--
Dillon Pyron                      | The opinions expressed are those of the
TI/DSEG Lewisville VAX Support    | sender unless otherwise stated.
(214)462-3556 (when I'm here)     |
(214)492-4656 (when I'm home)     |God gave us weather so we wouldn't complain
pyron@skndiv.dseg.ti.com          |about other things.
PADI DM-54909                     |

```

Sample Output:
```json
{
	"From": { "0": "pyron@skndiv.dseg.ti.com (Dillon Pyron)" },
	"Subject": { "0": "Re: Founding Father questions" },
	"Lines": { "0": "35" },
	"Nntp-Posting-Host": { "0": "skndiv.dseg.ti.com" },
	"Reply-To": { "0": "pyron@skndiv.dseg.ti.com" },
	"Organization": { "0": "TI/DSEG VAX Support" },
	"reply": {
		"0": " There is a veto proof majority in the house. The Senate, unfortunately, is a different story. The Lt.Gov. has vowed that the bill will not be voted on, and he has the power to do it. In addition, the Senate is a much smaller, and more readily manipulated body. On ther other hand, the semi-automatic ban will likely not live, as at least fifty per cent of the house currently opposes it, and it is VERY far down in the bill order in the Senate . And I thought my TX Political Science class was a waste of time! -- Dillon Pyron The opinions expressed are those of the TI/DSEG Lewisville VAX Support sender unless otherwise stated. 462-3556 492-4656 God gave us weather so we wouldn't complain about other things. PADI DM-54909 "
	},
	"reference_one": { "0": " Do people expect the Texans congressmen to act as the N.J. Republicans did? " },
	"reference_two": {
		"0": "Wasn't she the one making the comment in '88 about George being born with a silver foot in his mouth? Sounds like another damn politician to me. Ain't like the old days in Texas anymore. The politicians may have been corrupt then, but at least they'd take a stand. News now is that the House may already have a two-thirds majority, so her \"opposition\" out of her concern for image may not matter. "
	},
	"date": { "0": null },
	"delivered_to": { "0": [] },
	"to_domains": { "0": ["skndiv.dseg.ti.com"] },
	"error_message": { "0": null },
	"contained_emails": {
		"0": [" <1993Apr5.153951.25005@eagle.lerc.nasa.gov>, ", "pspod@bigbird.lerc.nasa.gov ", " >arc@cco.caltech.edu ", " pyron@skndiv.dseg.ti.com "]
	},
	"long_string": { "0": [] }
}
```
