import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from konlpy.tag import Okt
from konlpy.tag import Kkma
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from collections import Counterk

from soynlp import tokenizer
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences # padding
from tensorflow.keras.utils import to_categorical # one hot encoding
import pandas as pd
from sklearn.model_selection import train_test_split # data spliting



def test_text_preprocessing():


    print('📌 Text PreProcessing Summary:')

    nltk.download('punkt_tab')  # for tokenization
    nltk.download('averaged_perceptron_tagger_eng') # for Part-of-Speech
    nltk.download('stopwords') # for stopword
    nltk.download('wordnet') # for lemmatization

    # work_tokenize
    print('Word tokenization with NLTK word_tokenize :', word_tokenize(
        "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

    # WordPunctTokenizer
    print('Word tokenization with NLTK WordPunctTokenizer :', WordPunctTokenizer().tokenize(
        "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

    # TreebankWordTokenizer - retaining hypen words
    text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
    print('Word tokenization with NLTK TreebankWordTokenizer :', TreebankWordTokenizer().tokenize(text))

    # RegexpTokenizer - custom regex
    text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"
    tokenizer1 = RegexpTokenizer("[\w]+") # by number of char - more than 1
    tokenizer2 = RegexpTokenizer("\s+", gaps=True) # by whitespace
    print('Word tokenization with NLTK RegexpTokenizer :', tokenizer1.tokenize(text))
    print('Word tokenization with NLTK RegexpTokenizer :', tokenizer2.tokenize(text))

    # sent_tokenize
    text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
    print('Sentence tokenization with NLTK sent_tokenize :', sent_tokenize(text))

    # Part-of-Speech english - pos_tag
    text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
    tokenized_sentence = nltk.word_tokenize(text)
    print('Part-of-Speech tagging with pos_tag :', pos_tag(tokenized_sentence))

    # Part-of-Speech korean - okt, konlpy, kss, soyNLP
    # python 버전 mismatch 로 test skip
    # okt = Okt()
    # print('OKT Morpheme tokenization :', okt.morphs("새 맥북으로 하는 NLP, 재미있다."))
    # print('OKT Part-of-Speech tagging :', okt.pos("새 맥북으로 하는 NLP, 재미있다."))
    # print('OKT Noun extraction :', okt.nouns("새 맥북으로 하는 NLP, 재미있다."))

    # Cleaning
    text = "I was wondering if anyone out there could enlighten me on this car."
    short_word = re.compile(r'\W*\b\w{1,2}\b') # remove short words
    print("Cleaned word with re :", short_word.sub(' ', text))

    # stopword
    stop_words_list = stopwords.words('english')
    print("Number of stopword in nltk package :", len(stop_words_list))
    print("Stop words in nltk package :", stop_words_list[:20])

    example = "Family is not an important thing. It's everything."
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(example)

    result = []
    for word in word_tokens:
        if word not in stop_words:
            result.append(word)
    print("Before Filtering stopwords :", word_tokens)
    print("After Filtering stopwords :", result)


    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has',
             'starting']
    print("Before Lemmatization :", words)
    print("After Lemmatization :", [lemmatizer.lemmatize(word) for word in words])
    print(lemmatizer.lemmatize('dies'))
    print(lemmatizer.lemmatize('dies', 'v'))

    # Stemming
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer()
    sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."
    tokenized_sentence = nltk.word_tokenize(sentence)
    print("Before Stemming :", tokenized_sentence)
    print("After Porter Stemming :", [porter_stemmer.stem(word) for word in tokenized_sentence])
    print("After Lancaster Stemming :", [lancaster_stemmer.stem(word) for word in tokenized_sentence])


def test_encoding():

    print('📌 Integer Encoding :')

    # Integer Encoding
    raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."
    sentences = sent_tokenize(raw_text)
    print("Tokenized Sentence :", sentences)

    vocab = {}
    preprocessed_sentences = []
    stop_words = set(stopwords.words('english'))


    for sentence in sentences:
        # 1. sentence tokenization
        toknized_setence = word_tokenize(sentence)
        result = []

        # 2. word tokenization with Nomalization and Cleaning
        for word in toknized_setence:
            word = word.lower() # lower char
            if word not in stop_words: # filtering stopword
                if len(word) > 2: # filtering shorter than 3 char
                    result.append(word)
                    if word not in vocab: # counting
                        vocab[word] = 0
                    vocab[word] += 1
        preprocessed_sentences.append(result)


    ##### using dictionary method
    print('📌 Integer Encoding using Dictionary :')
    print("Normalized and Cleaned voca dictionary:", preprocessed_sentences)
    print("Counted Words:", vocab)

    # 3. Sorting dictionary by frequency
    sorted_vocab = sorted(vocab.items(), key=lambda x:x[1], reverse=True)
    print("Sorted Words by frequency:", sorted_vocab)

    # 4. indexing
    word_to_index = {}
    i = 0
    for word, freq in sorted_vocab:
        if freq > 1: # filtering low freq
            i = i + 1
            word_to_index[word] = i
    print("Indexed Words by frequency:", word_to_index)

    # 5. Cleaning except for Top 5
    vocab_size = 5
    words_frequency = [
        word for word, index in word_to_index.items()
        if index >= vocab_size + 1
    ]
    print("Out of Top 5 words:", words_frequency)

    # 6. removing OOT 5 words
    for word in words_frequency:
        del word_to_index[word]
    print("Top 5 dictionary :", word_to_index)

    # 7. integer encoding
    word_to_index['OOV'] = len(word_to_index) + 1
    print("Top 5 dictionary with OOV :", word_to_index)

    encoded_sentences = []
    for sentence in preprocessed_sentences:
        encoded_sentence = []
        for word in sentence:
            try:
                encoded_sentence.append(word_to_index[word])
            except KeyError:
                encoded_sentence.append(word_to_index['OOV'])
        encoded_sentences.append(encoded_sentence)
    print("Encoded preprocessed voca dictionary:", encoded_sentences)

    ##### using counter method
    print('📌 Integer Encoding using Counter :')
    print("Normalized and Cleaned voca dictionary:", preprocessed_sentences)
    print("Counted Words:", vocab)

    all_words_list = sum(preprocessed_sentences, [])
    print("All words list:", all_words_list)

    # removing redundant and counting by using Counter() moudle
    vocab = Counter(all_words_list)
    print("Vocab Count:", vocab)

    # removing OOT 7 words by using most_common()
    vocab_size = 7
    vocab = vocab.most_common(vocab_size)
    print("Vocab Count:", vocab)

    ##### using keras
    print('📌 Integer Encoding using Keras :')
    # Word tokenized sentences
    preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'],
                              ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'],
                              ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],
                              ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
                              ['barber', 'went', 'huge', 'mountain']]
    print("Word tokenized sentences with Normalization and Cleaning :", preprocessed_sentences)

    # Integer indexing by frequency
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(preprocessed_sentences) # frequency sorting
    print("Indexed words by frequency:", tokenizer.word_index)
    print("word count :", tokenizer.word_counts)
    print("Encoded sentences by frequency:", tokenizer.texts_to_sequences(preprocessed_sentences))

    # Cleaning except for Top 5
    voca_size = 5
    tokenizer = Tokenizer(num_words = voca_size + 1) # tokenizer setting reset
    tokenizer.fit_on_texts(preprocessed_sentences)
    print("Encoded words by frequency:", tokenizer.texts_to_sequences(preprocessed_sentences))

    # Considering OOV
    tokenizer = Tokenizer(num_words=vocab_size + 2, oov_token='OOV') # tokenizer setting reset
    tokenizer.fit_on_texts(preprocessed_sentences)
    # keras oov value is default 1
    print("Encoded words by frequency:", tokenizer.texts_to_sequences(preprocessed_sentences))



def test_padding():
    print('📌 Padding :')

    # numpay padding
    print('📌 Numpy Padding :')
    # Word tokenized sentences
    preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'],
                              ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'],
                              ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],
                              ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'],
                              ['barber', 'went', 'huge', 'mountain']]
    print("Word tokenized sentences with Normalization and Cleaning :", preprocessed_sentences)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(preprocessed_sentences) # frequency sorting
    encoded_sentences = tokenizer.texts_to_sequences(preprocessed_sentences)
    print("Encoded sentences by frequency:", encoded_sentences)

    # find max length word
    max_len = max(len(item) for item in encoded_sentences)
    print("max length :", max_len)

    # zero padding to length 7
    for sentence in encoded_sentences:
        while len(sentence) < max_len:
            sentence.append(0)

    padded_np = np.array(encoded_sentences)
    print("Padded sentences:", padded_np)

    # keras padding
    print('📌 Keras Padding :')
    encoded_sentences = tokenizer.texts_to_sequences(preprocessed_sentences)
    print("Encoded sentences by frequency:", encoded_sentences)

    # zero padding
    padded = pad_sequences(encoded_sentences,
                           padding='post', # filling zero behind
                           truncating='post', # cutting from behind
                           maxlen=5) # using dedicated shape
    print("Padded sentences:", padded)

def test_one_hot_encoding():
    ##### one hot encoding
    print('📌 One hot Encoding using Keras :')
    text = "Today We will have a good baseball game because We have many good baseball players. Baseball team We should fight"

    # 1. sentence tokenization
    sentences = sent_tokenize(text)
    print("After sentence tokenization", sentences)


    vocab = {}
    preprocessed_sentences = []
    stop_words = set(stopwords.words('english'))

    # 2. word tokenization / cleaning / normalization
    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)
        result = []

        for word in tokenized_sentence:
            word = word.lower() # Normalization lower char
            if word not in stop_words: # Cleaning stop word
                if len(word) > 2: # Cleaning short words
                    result.append(word)
                    if word not in vocab:
                        vocab[word] = 0
                    vocab[word] += 1
        preprocessed_sentences.append(result)
    print("Corpus After tokenization, cleaning, nomalization", preprocessed_sentences)
    print("vocab:", vocab)

    # 3. giving index by frequency with keras
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(preprocessed_sentences) # sorting
    print("vocabulary :", tokenizer.word_index)

    # 4. Integer encoding using vocabulary
    sub_text ="Also basketball players will have good good game today. Since We are one team and played mini game together"
    encoded_sub_text_sentences = tokenizer.texts_to_sequences([sub_text]) [0]
    print("Encoded sub_text:", encoded_sub_text_sentences)
    # [6, 2, 2, 4, 3, 7, 4]

    # 5. one-hot encoding
    one_hot_encoded_sub_text = to_categorical(encoded_sub_text_sentences)
    print("One hot encoded sub_text:", one_hot_encoded_sub_text)
    # [0. 0. 0. 0. 0. 0. 1. 0.] - one hot vector for idx 6
    # [0. 0. 1. 0. 0. 0. 0. 0.] - one hot vector for idx 2
    # [0. 0. 1. 0. 0. 0. 0. 0.] - one hot vector for idx 2
    # [0. 0. 0. 0. 1. 0. 0. 0.] - one hot vector for idx 4
    # [0. 0. 0. 1. 0. 0. 0. 0.] - one hot vector for idx 3
    # [0. 0. 0. 0. 0. 0. 0. 1.] - one hot vector for idx 7
    # [0. 0. 0. 0. 1. 0. 0. 0.] - one hot vector for idx 4

def test_data_splitting():

    print('📌 Data Splitting Summary:')

    print('📌 Total Dataset Splitting :')
    # dafaframe pandas
    values = [['Final Chance!', 1],
              ['Thanks for everything ...', 0],
              ['Dear June, It`s been a while...', 0],
              ['(AD) Don`t miss out this ..', 1]]
    columns = ['Content', 'Flag']

    df = pd.DataFrame(values, columns=columns)
    X = df['Content']
    y = df['Flag']
    print("X data :", X.tolist())
    print("y data :", y.tolist())

    # numpy
    np_array = np.arange(0, 16).reshape((4, 4))
    print("total numpy data :", np_array)

    X = np_array[:, :3]
    y = np_array[:, 3]
    print('X data :', X)
    print('y data :', y)

    print('📌 Test data Splitting :')

    # making arbitrary Total data X, y
    X, y = np.arange(10).reshape((5, 2)), range(5)
    print('Total X data :', X)
    print('Total y data :', list(y))

    # split each data as 7:3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    print('X_train data :', X_train)
    print('X_test data :', X_test)
    print('y_train data :', y_train)
    print('y_test data :', y_test)



















