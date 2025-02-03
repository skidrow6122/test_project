import torch
import tensorflow as tf
import keras

import sklearn
import gensim
import nltk
import konlpy

from src.services.data_analysis_example import (test_numpy, test_series, test_dataframe, test_matplotlib)
from src.services.text_preprocessing_example import (test_text_preprocessing)
from src.utils.file_loader import load_csv


def main():
    print("🚀 program stat")

    # library version definition
    print("PyTorch version:", torch.__version__)
    print("Tensorflow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print("Sklearn version:", sklearn.__version__)
    print("gensim version:", gensim.__version__)
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
    test_text_preprocessing()

if __name__ == "__main__":
    main()