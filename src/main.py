import keras
import konlpy
import nltk
import gensim
import sklearn
import tensorflow as tf
import torch

import numpy as np

from src.utils.file_loader import (load_csv)
from src.services.data_analysis_example import (test_numpy, test_series, test_dataframe, test_matplotlib )
from src.services.text_preprocessing_example import (test_text_preprocessing, test_encoding, test_padding, test_one_hot_encoding, test_data_splitting)
from src.services.count_based_word_representaion_example import (test_okt_bow, test_sklearn_bow, test_sklearn_tf_idf)
from src.services.cosine_similarity_example import (test_cos_sim, test_sklearn_cos_sim)
from src.services.machine_learning_example import (test_simple_differential, test_tensorflow_linear_regression,
                                                   test_keras_linear_regression, test_keras_logistic_regression,
                                                   test_keras_multi_input_linear_regression,
                                                   test_keras_multi_input_logistic_regression,
                                                   test_tensor_dimension)

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

    print('✅ NLP basic test - #1')
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

    print('✅ Text Preprocessing test - #2')
    # text processing test
    # test_text_preprocessing()

    # Encoding test
    #test_encoding()

    # Padding test
    #test_padding()

    # One hot encodig test
    # test_one_hot_encoding()

    # data splitting test
    # test_data_splitting()

    print('✅ Count based word Representation test - #4')
    #test_okt_bow()

    #test_sklearn_bow()

    #test_sklearn_tf_idf()

    print('✅ Cosine similarity test - #5')
    #test_cos_sim()

    #test_sklearn_cos_sim()

    print('✅ Machine learning test - #6')
    #test_simple_differential()

    #test_tensorflow_linear_regression()

    #test_keras_linear_regression()

    #test_keras_logistic_regression()

    #test_keras_multi_input_linear_regression()

    #test_keras_multi_input_logistic_regression()

    test_tensor_dimension()





if __name__ == "__main__":
    main()