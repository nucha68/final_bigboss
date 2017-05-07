
#기계학습의 목표: 분류와 회귀의 두 가지를 해결
#기계학습의 단계: 학습과 추론의 두 단계를 거침
#신경망 단계: 훈련데이터(학습데이터)를 사용해 가중치 매개변수를 학습, 앞에서 학습한 매개변수를 사용하여 입력데이터를 분류

import numpy as np
import matplotlib.pylab as plt

#Sigmoid Function
def sigmoid_func(x):
    return 1/(1 + np.exp(-x))

#Step Function
def step_func(x):
    return np.array(x > 0, dtype=np.int)

#Rectified Linear Unit Function
def relu_func(x):
    return np.maximum(0, x)

#Identity Function
def identity_func(x):
    return x

#Softmax Function - Warning: OverFlow 문제를 가지고 있음
def softmax_func_overflow(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#Softmax Function - Warning: OverFlow
#-중요 성질: 다음 성질 때문에 확률로 해석할 수 있음
#0.0에서 1.0 사이의 값을 가짐.
#총합은 1임.
def softmax_func(a):
    c = np.max(a)

    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

#=======================================================================================================================
#가중치, 편향 초기화
def init_network():
    network = {} #Dictionary 저장소
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])

    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])

    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network

#입력 x를
#network에 저장된 가중치와 편향을 이용하여
# 출력으로 변환
def forward(network, x):
    W1 = network["W1"]
    W2 = network["W2"]
    W3 = network["W3"]

    b1 = network["b1"]
    b2 = network["b2"]
    b3 = network["b3"]

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid_func(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid_func(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_func(a3)

    return y
#=======================================================================================================================

def run():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)

    print(y)

# if __name__ == "__main__":
#     run()

a = np.array([1010, 1000, 990])
y = softmax_func(a)
print(y)

b = np.array([0.3, 2.9, 4.0])
y = softmax_func(b)
print(y)