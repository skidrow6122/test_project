import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def test_numpy():
    vec = np.array([1,2,3,4,5])
    matrix = np.array([[10,20,30], [70,80,90]])
    print('📌 Numpy Summary:')
    print('vector:', vec)
    print('vector ndim:', vec.ndim)
    print('vector shape:', vec.shape)
    print('matrix :', matrix)
    print('matrix ndim:', matrix.ndim)
    print('matrix shape:', matrix.shape)

    slicing_mat = matrix[1, :] # slicing second row
    print('sliced matrix with second row : ', slicing_mat)
    print('picked value frm matrix :', matrix[1,2])

    # matrix, vector fill methods
    zero_mat = np.zeros((2,3)) # fill with 0
    print('zero matrix:', zero_mat)
    one_mat = np.ones((4,3)) # fill with 1
    print('one matrix:', one_mat)
    same_value_mat = np.full((5,5),7) # fill with same value
    print('same value matrix:', same_value_mat)
    eye_mat = np.eye(3) # fill with 1 diabonal line
    print('eye matrix:', eye_mat)
    random_mat = np.random.random((4,4)) # fill with random value
    print('random matrix:', random_mat)

    range_vec = np.arange(1,5)
    print('range vector:', range_vec)

    n=2
    step_range_vec = np.arange(1,10, n)
    print('step range vector:', step_range_vec)

    reshape_mat = np.array(np.arange(30)).reshape((5,6)) # transform the array
    print('reshape matrix:', reshape_mat)

    # numpy calculation
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print('add :', np.add(x,y))
    print('sub :', np.subtract(x,y))
    print('mult :', np.multiply(x,y))
    print('div :', np.divide(x,y))

    mat1 = np.array([[1, 2], [3, 4]])
    mat2 = np.array([[5, 6], [7, 8]])
    mat3 = np.dot(mat1, mat2) # matrix product
    print(mat3)


def test_series():
    series = pd.Series(["padres", "dodgers", "yankees", "mariners"],
                       index=[1, 2, 3, 4])

    print('📌 Series Summary:')
    print('-' * 15)
    print(series)
    print('Series Values:', series.values)
    print('Series Index:', series.index)


def test_dataframe():
    listData = [
        ['13', 'Manny', 0.287],
        ['23', 'Fernando', 0.309],
        ['9', 'Jake', 0.263]
    ]

    df = pd.DataFrame(listData, columns=['number', 'name', 'avg'])
    print("📌 DataFrame 테스트 1:")
    print(df)

    dictionaryData = {
        'number': ['13', '23', '9'],
        'name': ['Manny', 'Fernando', 'Jake'],
        'hr': [34, 46, 12]
    }

    df = pd.DataFrame(dictionaryData)
    print("📌 DataFrame 테스트 2:")
    print(df)


def test_matplotlib():
    plt.title('test')
    plt.plot([1,2,3,4,5],[2,4,9,8,10], label='Sample A')
    plt.plot([1.5,2.5,6,3,6],[4,2,4,6,7], label='Sample B')
    plt.xlabel('avg')
    plt.ylabel('number')
    plt.legend()
    plt.show()