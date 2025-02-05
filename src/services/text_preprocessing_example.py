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
from collections import Counter

from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


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
    shortword = re.compile(r'\W*\b\w{1,2}\b') # remove short words
    print("Cleaned word with re :", shortword.sub(' ', text))

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
    print("Encoded Preprocessed voca dictionary:", encoded_sentences)

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















