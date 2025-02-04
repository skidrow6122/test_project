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








