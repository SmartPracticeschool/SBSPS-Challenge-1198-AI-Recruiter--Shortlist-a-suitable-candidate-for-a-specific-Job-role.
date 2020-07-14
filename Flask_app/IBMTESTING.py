import textract
import os
from os import listdir
from os.path import isfile, join
from io import StringIO
import pandas as pd
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from spacy.matcher import PhraseMatcher



def pdfextract(file):
    text=textract.process(file, method='pdfminer')
    return text


# function that does phrase matching and builds a candidate profile
def create_profile(file):
    file1 = open(file, 'r+')
    text = file1.read()
    text = str(text)

    # below is the csv where we have all the keywords
    keyword_dict = pd.read_csv('template.csv')
    print('The templates on which assessment will be done \n\n\n', keyword_dict.head())

    # Creating Directory for individual words
    openness_words = [nlp(text) for text in keyword_dict['openness'].dropna(axis=0)]
    neuroticism_words = [nlp(text) for text in keyword_dict['neuroticism'].dropna(axis=0)]
    conscientiousness_words = [nlp(text) for text in keyword_dict['conscientiousness'].dropna(axis=0)]
    agreeableness_words = [nlp(text) for text in keyword_dict['agreeableness'].dropna(axis=0)]
    extraversion_words = [nlp(text) for text in keyword_dict['extraversion']]

    matcher = PhraseMatcher(nlp.vocab)
    matcher.add('Openness', None, *openness_words)
    matcher.add('Neuroticism', None, *neuroticism_words)
    matcher.add('Conscientiousness', None, *conscientiousness_words)
    matcher.add('Agreeableness', None, *agreeableness_words)
    matcher.add('Extraversion', None, *extraversion_words)
    doc = nlp(text)

    d = []
    print(d)
    matches = matcher(doc)

    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # get the unicode ID, i.e. 'COLOR'
        span = doc[start: end]  # get the matched slice of the doc
        d.append((rule_id, span.text))

    keywords = "\n".join(f'{i[0]} {i[1]} ({j})' for i, j in Counter(d).items())

    ## convertimg string of keywords to dataframe
    df = pd.read_csv(StringIO(keywords), names=['Keywords_List'])
    df1 = pd.DataFrame(df.Keywords_List.str.split(' ', 1).tolist(), columns=['Subject', 'Keyword'])
    df2 = pd.DataFrame(df1.Keyword.str.split('(', 1).tolist(), columns=['Keyword', 'Count'])
    df3 = pd.concat([df1['Subject'], df2['Keyword'], df2['Count']], axis=1)
    df3['Count'] = df3['Count'].apply(lambda x: x.rstrip(")"))

    base = os.path.basename(file)
    filename = os.path.splitext(base)[0]

    name = filename.split('_')
    name2 = name[0]
    name2 = name2.lower()
    ## converting str to dataframe
    name3 = pd.read_csv(StringIO(name2), names=['Candidate Name'])

    dataf = pd.concat([name3['Candidate Name'], df3['Subject'], df3['Keyword'], df3['Count']], axis=1)
    dataf['Candidate Name'].fillna(dataf['Candidate Name'].iloc[0], inplace=True)

    return (dataf)

def main_process1(file):
    final_database = pd.DataFrame()

    dat = create_profile(file)
    final_database = final_database.append(dat)

    final_database2 = final_database['Keyword'].groupby([final_database['Candidate Name'], final_database['Subject']]).count().unstack()
    final_database2.fillna(0,inplace=True)

    new_data = final_database2.iloc[:,1:]
    list1=['Openness', 'Neuroticism ', 'Conscientiousness','Agreeableness','Extraversion']

    for i in list1:
        if i not in final_database2.columns[1:]:
            final_database2[i] = 0
    print(final_database2)
    return final_database2