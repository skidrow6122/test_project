import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from konlpy.tag import Okt
from konlpy.tag import Kkma


def test_text_preprocessing():

    print('📌 Text PreProcessing Summary:')

    nltk.download('punkt_tab')  # for tokenization
    nltk.download('averaged_perceptron_tagger_eng') # for Part-of-Speech

    # work_tokenize
    print('Word tokenization with NLTK word_tokenize :', word_tokenize(
        "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

    # WordPunctTokenizer
    print('Word tokenization with NLTK WordPunctTokenizer :', WordPunctTokenizer().tokenize(
        "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))

    # TreebankWordTokenizer - retaining hypen words
    text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
    print('Word tokenization with NLTK TreebankWordTokenizer :', TreebankWordTokenizer().tokenize(text))

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







