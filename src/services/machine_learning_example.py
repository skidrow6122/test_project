import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers


def test_simple_differential():
    print('##### tensorflow simple automatic differential test')

    w = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        z = f(w)
        print("z", z.numpy())

    # 일반적인 방정식에 대한 미분
    # 방정식 계산값을 파라미터로 넣음 z = 13, 하지만 tape 안에는 z 에 해당하는 함수식이 바인딩 되어있음
    gradients = tape.gradient(z, [w])
    print("simple differential test gradient :", gradients)

# 2w^2 + 5
def f(w):
    y = w ** 2
    z = 2*y +5
    return z


def test_tensorflow_linear_regression():
    print('##### tensorflow linear regression test')

    # 초기 파라미터
    w = tf.Variable(4.0)
    b = tf.Variable(1.0)

    @tf.function  # enhanced performance by using GPU/TPU
    def hypothesis(x):
        return w * x + b

    x_test = [3.5, 5, 5.5, 6]
    print(hypothesis(x_test).numpy()) # transforming to numpy array format

    # Mean squered error
    @tf.function
    def mse_loss_function(y_pred, y):
        return tf.reduce_mean(tf.square(y_pred - y))

    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [11, 22, 33, 44, 53, 66, 77, 87, 95]

    # Stochastic gradient descent
    optimizer = tf.optimizers.SGD(learning_rate=0.01)

    for i in range(201):
        with tf.GradientTape() as tape:
            y_pred = hypothesis(x) # 가설함수에 x 데이터를 넣어 y 예측치 계산
            cost = mse_loss_function(y_pred, y) # loss function으로 평균제곱오차 값 계산
        # 평균제곱오차를 미분
        grads = tape.gradient(cost, [w, b])
        #print("gradient of cost :", grads) # cost 를 w와 b에 대해 편미분한 결과

        # 파라미터 업데이트
        optimizer.apply_gradients(zip(grads, [w, b])) # 기울기와 파라미터들을 쌍으로 묶음

        if i % 10 == 0:
            print("epoch : {:3} | w의 값 : {:5.4f} | b의 값 : {:5.4f} | cost : {:5.4f}".format(i, w.numpy(), b.numpy(), cost))

    x_test = [3.5, 5, 5.5, 6]
    print("tensorflow after learning :", hypothesis(x_test).numpy())


def test_keras_linear_regression():
    print('##### keras linear regression test')

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([11, 22, 33, 44, 53, 66, 77, 87, 95])

    model = Sequential()

    # 모델 정의
    # 출력 y의 차원1, 입력 x의 차원1, 선형회귀 이므로 activation 은 linear
    # 이렇게 정의하면 hypothesis() 가설함수를 선형회귀식인 w * x + b 로 자동 생성
    model.add(Dense(1, input_dim=1, activation='linear'))

    # 옵티마이저 정의
    sgd = optimizers.SGD(learning_rate=0.01)

    # loss function 정의
    model.compile(optimizer=sgd, loss='mse', metrics=['mse'])

    # 학습 수행 - 이 시점에 weight 와 bias를 랜덤 초기화
    model.fit(x, y, epochs=300)

    # 시각화 : 예측값은 블루라인으로, 실제 값은 검은 점으로
    plt.plot(x, model.predict(x), 'b', label="Predicted Line")  # 모델 예측값
    plt.plot(x, y, 'k.', label="Actual Data")  # 실제 데이터
    plt.xlabel("Effort Time")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    # 학습 완료된 모델로 입력 데이터에 대한 예측
    print("keras after learning linear regression:", model.predict(np.array([9.5])))


def test_keras_logistic_regression():
    print('##### keras logistic regression test')

    # 독립 변수
    x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
    # 종속 변수. x 가 10이상일 경우 1, 미만일 경우 0을 부여한 레이블
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).reshape(-1, 1)

    model = Sequential()

    # 모델 정의
    model.add(Dense(1, input_dim=1, activation='sigmoid'))

    # 옵티마이저 정의
    sgd = optimizers.SGD(learning_rate=0.01)

    # loss function 정의
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_crossentropy'])

    # 학습 수행 - 이 시점에 weight 와 bias를 랜덤 초기화
    model.fit(x, y, epochs=200)

    # 시각화 : 예측값은 블루라인으로, 실제 값은 검은 점으로
    plt.plot(x, model.predict(x), 'b', label="Predicted Line")  # 모델 예측값
    plt.plot(x, y, 'k.', label="Actual Data")  # 실제 데이터
    plt.xlabel("Effort Time")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    # 학습 완료된 모델로 입력 데이터에 대한 예측
    print("keras after learning logistic regression:",model.predict(np.array([1, 2, 3, 4, 4.5, 6, 11, 42, 500])))

def test_keras_multi_input_linear_regression():
    print('##### keras multi input linear regression test')

    # 학습 데이터 정의
    X = np.array([[70,85,11], [71,89,18], [50,80,20], [99,20,10], [50,10,10]])
    y = np.array([73, 82 ,72, 57, 34]) # 최종 성적

    model = Sequential()
    model.add(Dense(1, input_dim=3, activation='linear'))
    sgd = optimizers.SGD(learning_rate=0.0001)
    model.compile(optimizer=sgd, loss='mse', metrics=['mse'])
    model.fit(X,y, epochs=2000)
    # 학습 완료된 model 로 검증
    print(model.predict(X))

    # 테스트 데이터 정의
    X_test = np.array([[20,99,10], [40,50,20]])
    # 테스트
    print(model.predict(X_test))

def test_keras_multi_input_logistic_regression():
    print('##### keras multi input logistic regression test')

    # 학습 데이터 정의
    X = np.array([[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]])
    y = np.array([0, 0, 0, 1, 1, 1]) #최종 성적

    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='sigmoid'))
    sgd = optimizers.SGD(learning_rate=0.01)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.fit(X, y, epochs=2000)
    print(model.predict(X))

def test_tensor_dimension():
    print('##### tensor dimension test')










