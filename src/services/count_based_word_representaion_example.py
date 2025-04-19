from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def test_okt_bow():
    print('📌 Bag of Words test')

    okt = Okt() # okt 모듈은 python 3.10 이하 버전에서만 쓸 것
    document = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
    # 마침표 제거 및 형태소 분석
    document = document.replace('.', '')
    tokenized_document = okt.morphs(document)

    word_to_index = {}
    bow = []

    for word in tokenized_document:
        if word not in word_to_index.keys():
            word_to_index[word] = len(word_to_index)
            # bow에 기본값 1 세팅
            bow.insert(len(word_to_index) - 1,1)
        else:
            # 재등장하는 단어의 인덱스
            index = word_to_index.get(word)
            # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더해줌
            bow[index] = bow[index] + 1

    print("BoW vocabulary:", word_to_index)
    print("BoW vector:", bow)

def test_sklearn_bow():
    print('📌 Bag of Words test')

    corpus = ['you know I want your love. because I love you.']
    # vector = CountVectorizer(stop_words='english')
    vector = CountVectorizer()

    # count frequency from corpus
    print("sklearn BoW vector:", vector.fit_transform(corpus).toarray())
    print("sklearn BoW vocabulary:", vector.vocabulary_)


def test_sklearn_tf_idf():
    print('📌 TF-IDF test')

    corpus = ['you know I want your love',
               'I like you',
               'what should I do',
    ]

    vector = CountVectorizer()

    # count frequency from corpus : actually it is same to DTM
    print("sklearn DTM vector:", vector.fit_transform(corpus).toarray())
    print("sklearn DTM vocabulary:", vector.vocabulary_)

    # tf-idf calculating
    tfidfv = TfidfVectorizer().fit(corpus)
    print("sklearn TF-IDF vector:", tfidfv.transform(corpus).toarray())
    print("sklearn TFIDF vocabulary:", tfidfv.vocabulary_)





