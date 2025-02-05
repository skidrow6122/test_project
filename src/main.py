import keras
import konlpy
import nltk
import gensim
import sklearn
import tensorflow as tf
import torch

import numpy as np

from src.services.text_preprocessing_example import (test_encoding)

def main():
    print("🚀 program start")

    # library version definition
    print("PyTorch version:", torch.__version__)
    print("Tensorflow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print("Gensim version:", gensim.__version__)
    print("Numpy version:", np.__version__)
    print("Sklearn version:", sklearn.__version__)
    print("NLTK version:", nltk.__version__)
    print("konlpy version:", konlpy.__version__)

    # numpy test
    #test_numpy()

    # Series test
    #test_series()

    # DataFrame test
    #test_dataframe()

    # CSV load test
    #df = load_csv('zipcode_driver.csv')
    #print(df)

    # Visualization test
    #test_matplotlib()

    # text processing test
    #test_text_preprocessing()

    # Encoding test
    test_encoding()



if __name__ == "__main__":
    main()