import pandas as pd
import nltk
from nltk.corpus import stopwords
lemmatizer = nltk.stem.WordNetLemmatizer()
stop = stopwords.words('english')
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.set_option('max_rows', 50)
# Reading the data from the .tsv file
fullcorpus=pd.read_table('D:\\7th semester\\NLP\\SMSSpamCollection.tsv', sep="\t", header=None,names=['label', 'sms_message'])
# Removing punctuation
fullcorpus["clean_message"] = fullcorpus['sms_message'].str.replace('[^\w\s]','')
# Tokenizing message dataset
fullcorpus["tokenized_message"]=fullcorpus['clean_message'].apply(lambda x:nltk.word_tokenize(x.lower()))
# Removing stopwords
fullcorpus["message_without_stopword"]=fullcorpus['tokenized_message'].apply(lambda x: [item for item in x if item not in stop])

# Lemmatization of data
fullcorpus['lematized_message']=fullcorpus['message_without_stopword'].apply(lambda row: [lemmatizer.lemmatize(word) for word in row])
# Creating a bigram dataset
fullcorpus['bigram']=fullcorpus['lematized_message'].apply(lambda x: list(nltk.ngrams(x, 2)))
fullcorpus['ham_corpus']=fullcorpus[fullcorpus.label=='ham']['bigram']
fullcorpus['spam_corpus']=fullcorpus[fullcorpus.label=='spam']['bigram']
# print(fullcorpus.head())
ham_bigram=[]
spam_bigram=[]
spam_unigram=[]
ham_unigram=[]
for w in fullcorpus[fullcorpus.label=='ham']['bigram']:
     ham_bigram=ham_bigram+w
for w in fullcorpus[fullcorpus.label=='spam']['bigram']:
    spam_bigram=spam_bigram+w
for w in fullcorpus[fullcorpus.label=='spam']['lematized_message']:
    spam_unigram=spam_unigram+w
for w in fullcorpus[fullcorpus.label=='ham']['lematized_message']:
    ham_unigram=ham_unigram+w

def create_bigramdictionery(bigramcorpus):
    bidictionery = {}
    for x in bigramcorpus:
        if x not in bidictionery:
            bidictionery[x] =1
        else:
            bidictionery[x]+=1
    return bidictionery

def create_unigramdictionery(unigramcorpus):
    unidictionery = {}
    for x in unigramcorpus:
        if x not in unidictionery:
            unidictionery[x] = 1
        else:
            unidictionery[x] += 1
    return unidictionery

spam_bidict=create_bigramdictionery(spam_bigram)
ham_bidict=create_bigramdictionery(ham_bigram)
spam_unidict=create_unigramdictionery(spam_unigram)
ham_unidict=create_unigramdictionery(ham_unigram)

def find_bigram(target_message):
    target_message = nltk.re.sub(r'[^\w\s]', '', target_message)
    word_tokens = nltk.word_tokenize(target_message.lower())
    filtered_sentence = [w for w in word_tokens if not w in stop]
    message = " ".join(filtered_sentence)
    word_tokens = nltk.word_tokenize(message)
    lemmatized_message = []
    p = 1
    for ms in word_tokens:
        lemmatized_message.append(nltk.WordNetLemmatizer().lemmatize(ms))
    bigram_message=list(nltk.ngrams(lemmatized_message, 2))
    return bigram_message

def calculate_probability(first_word,second_word,unigram_model,bigram_model):
    V = len(unigram_model)
    try:
        N = unigram_model[first_word]
    except KeyError:
        N = 0
    try:
        C = bigram_model[first_word, second_word]
    except KeyError:
        C = 0
    prob = (C + 1) / (N + V)
    return prob


message1="Sorry, ..use your brain dear"
message1_bigram=find_bigram(message1)
message1_ham_prob=1
for x in message1_bigram:
    message1_ham_prob=message1_ham_prob*calculate_probability(message1_bigram[0],message1_bigram[1],ham_unidict,ham_bidict)
print('Ham dataset probability for message 1 -',message1_ham_prob)

message1_spam_prob=1
for x in message1_bigram:
    message1_spam_prob=message1_spam_prob*calculate_probability(message1_bigram[0],message1_bigram[1],spam_unidict,spam_bidict)
print('Spam dataset probability for message 1 -',message1_spam_prob)

message2=" SIX chances to win CASH."
message2_bigram=find_bigram(message2)
message2_ham_prob=1
for x in message2_bigram:
    message2_ham_prob=message2_ham_prob*calculate_probability(message2_bigram[0],message2_bigram[1],ham_unidict,ham_bidict)
print('Ham dataset probability for message 2 -',message2_ham_prob)

message2_spam_prob=1
for x in message2_bigram:
    message2_spam_prob=message2_spam_prob*calculate_probability(message2_bigram[0],message2_bigram[1],spam_unidict,spam_bidict)
print('Spam dataset probability for message 2 -',message2_spam_prob)


