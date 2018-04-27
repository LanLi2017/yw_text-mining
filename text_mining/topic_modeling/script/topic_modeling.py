# @begin topic_modeling @desc using NLTK and YW to do Topic_modeling
# @in Static_text @uri file:news2012.txt
# @param stopwords
# @param regexr @as regular_expression
# @param num_topics
# @param num_iteration
# @out run_log  @desc record the variable data for all the events occur in the procedure
# @out LDA_model  @desc using LDA model to do topic modeling @uri file:LDA_model.txt
import os
from gensim import corpora,models
from collections import defaultdict

from gensim.models import Phrases
from nltk import pprint,RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords

# @begin initialize_run @desc Using Logging to track events that happen when software runs
# @out run_log @uri file:run_log.log
import logging
logging.basicConfig(filename='run_log.log',format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
# @end initialize_run

# @begin ConstructCorpus @desc Using Gensim to construct corpus to train LDA models
# @in Static_text @desc Text file which contains static text data @uri file:news2012.txt
# @param stopwords
# @param regexr @as regular_expression
# @out dictionary @as dictionary @desc a mapping between words and their integer ids
# @out corpus @as Corpus


# @begin AccessText @desc To read the text data
# @in Static_text @uri file:news2012.txt
# @out text_file @as TextRead
data_dir = '../../nipstxt/'

yr = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
dir = ['nips' + year for year in yr]


# Read all texts into a list.
text_file = []
for yr_dir in dir:
    files = os.listdir(data_dir + yr_dir)
    for filen in files:
        # Note: ignoring characters that cause encoding errors.
        with open(data_dir + yr_dir + '/' + filen, errors='ignore') as fid:
            txt = fid.read()
        text_file.append(txt)
# @end AccessText

# @begin PreprocessFile @desc To preprocess the text data
# @param stopwords
# @param regular_expression
# @in TextRead
# @out dictionary
# @out Corpus
en_stopwords=set(stopwords.words('english'))
de_stopwords=set(stopwords.words('german'))
fr_stopwords=set(stopwords.words('french'))
stopwords=en_stopwords|de_stopwords|fr_stopwords

tokenizer = RegexpTokenizer(r'\w+')
for idx in range(len(text_file)):
    text_file[idx]=text_file[idx].lower()
    text_file[idx]=tokenizer.tokenize(text_file[idx])

# remove numeric words
text_file=[[text for text in doc if not text.isnumeric()]for doc in text_file]

# remove stop words
text_file=[[text for text in doc if text not in stopwords]for doc in text_file]
# calculate word frequencies
word_frequency=defaultdict(int)
for text in text_file:
    for token in text:
        word_frequency[token] +=1

processed_corpus=[[token for token in text if word_frequency[token]>1] for text in text_file]
# Lemmatize all words in documents.
lemmatizer = WordNetLemmatizer()
processed_corpus = [[lemmatizer.lemmatize(token) for token in doc] for doc in processed_corpus]

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(processed_corpus, min_count=20)
for idx in range(len(processed_corpus)):
    for token in bigram[processed_corpus[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            processed_corpus[idx].append(token)

# associate each word in the corpus with a unique integer ID
dictionary=corpora.Dictionary(processed_corpus)
dictionary.filter_extremes(no_below=20, no_above=0.5)
corpus1= [dictionary.doc2bow(text) for text in processed_corpus]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus1))
# @end PreprocessFile

# @end ConstructCorpus

# @begin Train_LDA_Model @desc using Gensim.LDA to do training
# @in Corpus
# @in dictionary
# @param num_iteration
# @param num_topics
# @out LDA_model @desc Transformation from BOW counts into a topic space @uri file:LDA_model.txt

_num_topics=20
num_iteration=500
id2word = dictionary.id2token

with open('LDA_model.txt','wt')as f1:
    LDA_model=models.LdaModel(corpus=corpus1,id2word=dictionary,num_topics=_num_topics,iterations=num_iteration,passes=20)
    top_topics=LDA_model.top_topics(corpus1,topn=20)
    pprint(top_topics)
    for topic in top_topics:
        f1.write(str(topic)+'\n')

# @end Train_LDA_Model
# @end topic_modeling
