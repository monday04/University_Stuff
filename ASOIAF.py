#                    IMPORT PACKAGES                  #
import codecs
import re
import collections

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

# nltk.download('stopwords')
from nltk.corpus import stopwords


#                     READING DATA                    #
with codecs.open('/Users/zethlenedelosreyes/Desktop/Programming/Python/1_AGOT.txt', 'r', 'utf-8') as f:
    text_AGOT = f.read()
with codecs.open('/Users/zethlenedelosreyes/Desktop/Programming/Python/2_ACOK.txt', 'r', 'utf-8') as f:
    text_ACOK = f.read()
with codecs.open('/Users/zethlenedelosreyes/Desktop/Programming/Python/3_ASOS.txt', 'r', 'utf-8') as f:
    text_ASOS = f.read()
with codecs.open('/Users/zethlenedelosreyes/Desktop/Programming/Python/4_AFFC.txt', 'rb') as f:
    text_AFFC = f.read().decode(errors='replace')
with codecs.open('/Users/zethlenedelosreyes/Desktop/Programming/Python/5_ADWD.txt', 'rb') as f:
    text_ADWD = f.read().decode(errors='replace')
with codecs.open('/Users/zethlenedelosreyes/Desktop/Programming/Python/CharNames.txt', 'r', 'utf-8') as f:
    text_cnames = f.readlines()


#                   PROCESSING DATA                   #
# Checking for Stopwords
esw = stopwords.words('english')
esw.append("would")
esw.append("could")
esw.append("said")

# Filter tokens using regular expressions
word_pattern = re.compile("^\w+$")


# create a token counter function
def get_text_counter(text):
    tokens = WordPunctTokenizer().tokenize(PorterStemmer().stem(text))
    tokens = list(map(lambda x: x.lower(), tokens))
    tokens = [token for token in tokens if re.match(word_pattern, token) and token not in esw]
    return collections.Counter(tokens)


# Create a function to calculate the absolute frequency of the most common words
def make_df(counter):
    abs_freq = np.array([el[1] for el in counter])
    index = [el[0] for el in counter]
    df = pd.DataFrame(data=np.array([abs_freq]).T, index=index,
                      columns=["Absolute Frequency"])
    df.index.name = "Most Common Words"
    return df


# format the name list
text_cnames = [i.strip() for i in text_cnames]


# Create a function to extract character names from the most common words
def top_char(list, book_char):
    temp_var = []
    for line in list:
        for character in text_cnames:
            if character in line:
                if character == 'daenerys':
                    temp_var = line
                if character == 'dany':
                    line[0] = 'daenerys'
                    if temp_var:
                        line[1] = line[1] + temp_var[1]
                book_char.append(line)
    return book_char


#                   ANALYSIS                    #
#           Analyze Individual Texts            #

# Calculate the most common words from each book and extract the most popular characters
AGOT_counter = get_text_counter(text_AGOT)
AGOT_df = make_df(AGOT_counter.most_common(250))
AGOT_list = [AGOT_df.columns.tolist()] + AGOT_df.reset_index().values.tolist()
AGOT_char = top_char(AGOT_list, [])

ACOK_counter = get_text_counter(text_ACOK)
ACOK_df = make_df(ACOK_counter.most_common(250))
ACOK_list = [ACOK_df.columns.tolist()] + ACOK_df.reset_index().values.tolist()
ACOK_char = top_char(ACOK_list, [])

ASOS_counter = get_text_counter(text_ASOS)
ASOS_df = make_df(ASOS_counter.most_common(250))
ASOS_list = [ASOS_df.columns.tolist()] + ASOS_df.reset_index().values.tolist()
ASOS_char = top_char(ASOS_list, [])

AFFC_counter = get_text_counter(text_AFFC)
AFFC_df = make_df(AFFC_counter.most_common(250))
AFFC_list = [AFFC_df.columns.tolist()] + AFFC_df.reset_index().values.tolist()
AFFC_char = top_char(AFFC_list, [])

ADWD_counter = get_text_counter(text_ADWD)
ADWD_df = make_df(ADWD_counter.most_common(250))
ADWD_list = [ADWD_df.columns.tolist()] + ADWD_df.reset_index().values.tolist()
ADWD_char = top_char(ADWD_list, [])


# Create a copy of results to CSV
np.savetxt("ACOK_list.csv", ACOK_char, delimiter=",", fmt='%s')
np.savetxt("AGOT_list.csv", AGOT_char, delimiter=",", fmt='%s')
np.savetxt("AFFC_list.csv", AFFC_char, delimiter=",", fmt='%s')
np.savetxt("ASOS_list.csv", ASOS_char, delimiter=",", fmt='%s')
np.savetxt("ADWD_list.csv", ADWD_char, delimiter=",", fmt='%s')


#               PLOTTING NAME FREQUENCIES            #
#           Plotting Names from Individual Books           #


def plot_var(book):
    x, y = [], []
    for line in book:
        x.append(line[0])
    for line in book:
        y.append(line[1])

    return x, y


x, y = plot_var(AGOT_char)
plt.barh(x, y, color='red')
plt.xlabel('Name Occurences')
plt.title('Book1: A Game Of Thrones')
plt.show()

x, y = plot_var(ACOK_char)
plt.barh(x, y, color='red')
plt.xlabel('Name Occurences')
plt.title('Book2: A Clash of Kings')
plt.show()

x, y = plot_var(ASOS_char)
plt.barh(x, y, color='red')
plt.xlabel('Name Occurences')
plt.title('Book3: A Storm of Swords')
plt.show()

x, y = plot_var(AFFC_char)
plt.barh(x, y, color='red')
plt.xlabel('Name Occurences')
plt.title('Book4: A Feast for Crows')
plt.show()

x, y = plot_var(ADWD_char)
plt.barh(x, y, color='red')
plt.xlabel('Name Occurences')
plt.title('Book5: A Dance With The Dragons')
plt.show()
