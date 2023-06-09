"""
A module for storing text-preprocessing functionality.
Resposible for cleaning up the  unnecessary characters/noise from text
"""
# REPLACED 
# import jsonlines --> import json (import jsonlines doesn't exist???)
import json
import os
import pandas as pd
import pickle
import re

# ADDED
# to accept command-line args for bash script (cleanText.sh)
import sys

def replace_semicolon(text, threshold=10):
    '''
    Get rid of semicolons.
    First split text into fragments between the semicolons. If the fragment 
    is longer than the threshold, turn the semicolon into a period. O.w treat
    it as a comma.
    Returns new text
    '''
    new_text = ""
    for subset in re.split(';', text):
        subset = subset.strip() # Clear off spaces
        # Check word count
        if len(subset.split()) > threshold:
            # Turn first char into uppercase
            new_text += ". " + subset[0].upper() + subset[1:]
        else:
            # Just append with a comma 
            new_text += ", " + subset

    return new_text

USC_re = re.compile('[Uu]\.*[Ss]\.*[Cc]\.]+')
PAREN_re = re.compile('\([^(]+\ [^\(]+\)')
BAD_PUNCT_RE = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
BULLET_RE = re.compile('\n[\ \t]*`*\([a-zA-Z0-9]*\)')
DASH_RE = re.compile('--+')

# modified whitespace_re to not include the \n\n case
# so it should cover extra tabs and \n, but not \n\n
WHITESPACE_RE = re.compile('(?<!\n)[\t\n](?!\n)')

# ADDED
# to match on new paragraphs which is "\n\n" I think 
NEW_PARAGRAPH_RE = re.compile('\n\n')

# ADDED
# to match on an annoying error with way too many unnecessary random spaces
TOO_MANY_SPACES = re.compile('[^\S\r\n]+')

EMPTY_SENT_RE = re.compile('[,\.]\ *[\.,]')
FIX_START_RE = re.compile('^[^A-Za-z]*')
FIX_PERIOD = re.compile('\.([A-Za-z])')
SECTION_HEADER_RE = re.compile('SECTION [0-9]{1,2}\.|\nSEC\.* [0-9]{1,2}\.|Sec\.* [0-9]{1,2}\.')

FIX_PERIOD = re.compile('\.([A-Za-z])')

SECTION_HEADER_RE = re.compile('SECTION [0-9]{1,2}\.|\nSEC\.* [0-9]{1,2}\.|Sec\.* [0-9]{1,2}\.')

def clean_text(text):
    """
    Borrowed from the FNDS text processing with additional logic added in.
    Note: we do not take care of token breaking - assume SPACY's tokenizer
    will handle this for us.
    """

    # Indicate section headers, we need them for features
    text = SECTION_HEADER_RE.sub('SECTION-HEADER', text)
    # For simplicity later, remove '.' from most common acronym
    text = text.replace("U.S.", "US")
    text = text.replace('SEC.', 'Section')
    text = text.replace('Sec.', 'Section')
    text = USC_re.sub('USC', text)

    # Remove parantheticals because they are almost always references to laws 
    # We could add a special tag, but we just remove for now
    # Note we dont get rid of nested parens because that is a complex re
    #text = PAREN_re.sub('LAWREF', text)
    text = PAREN_re.sub('', text)
    

    # Get rid of enums as bullets or ` as bullets
    text = BULLET_RE.sub(' ',text)
    
    # Clean html 
    text = text.replace('&lt;all&gt;', '')

    # Remove annoying punctuation, that's not relevant
    text = BAD_PUNCT_RE.sub('', text)

    # Get rid of long sequences of dashes - these are formating
    text = DASH_RE.sub( ' ', text)

    # ADDED
    # remove stupid spaces, why does this even happen
    text = TOO_MANY_SPACES.sub(' ', text)

    # removing newlines, tabs, and extra spaces.
    text = WHITESPACE_RE.sub('', text)

    # ADDED 
    # remove \n\n (new paragraphs) and replace with \t
    text = NEW_PARAGRAPH_RE.sub('\t', text)

    # If we ended up with "empty" sentences - get rid of them.
    text = EMPTY_SENT_RE.sub('.', text)
    
    # Attempt to create sentences from bullets 
    text = replace_semicolon(text)
    
    # Fix weird period issues + start of text weirdness
    text = re.sub('\.(?=[A-Z])', '  . ', text)
    # Get rid of anything thats not a word from the start of the text
    text = FIX_START_RE.sub( '', text)
    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent   
    text = FIX_PERIOD.sub(". \g<1>", text)

    # Fix quotes
    text = text.replace('``', '"')
    text = text.replace('\'\'', '"')

    # ADDED 
    # idk why but this fixes the multiple spaces issue?
    text = text.replace('  ', ' ')

    # Add special punct back in
    text = text.replace('SECTION-HEADER', '<SECTION-HEADER>')

    return text


if __name__ == '__main__':

    # opening the input (bills) and output (cleaned text) files
    # bills = open("./data/billsToClean.jsonl", "r")
    bills = open(sys.argv[1], "r")

    # cleaned = open("./data/cleanText.txt", "w")
    cleaned = open(sys.argv[2], "w")

    # iterating through each line in bills, since each line is a JSON object
    for line in bills:
        data = json.loads(line)
        # use the modified clean_text function, append a \n to the end
        # so it works with our format
        cleaned.write(f'{clean_text(data["text"])}\n')

    # closing each file
    bills.close()
    cleaned.close()
