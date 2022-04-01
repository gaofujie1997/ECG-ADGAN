import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

X_N = pickle.load(open("../data/X_AMMI_N.pkl", "rb"))[0:1000]
print(np.shape(X_N))
X_N_label = np.zeros([np.shape(X_N)[0], 1])

X_S = pickle.load(open("../data/X_AMMI_S.pkl", "rb"))[0:330]
print(np.shape(X_S))
X_S_label = np.zeros([np.shape(X_S)[0], 1]) + 1

X_V = pickle.load(open("../data/X_AMMI_V.pkl", "rb"))[0:330]
print(np.shape(X_V))
X_V_label = np.zeros([np.shape(X_V)[0], 1]) + 1

X_F = pickle.load(open("../data/X_AMMI_F.pkl", "rb"))[0:330]
print(np.shape(X_F))
X_F_label = np.zeros([np.shape(X_F)[0], 1]) + 1

X_Q = pickle.load(open("../data/X_AMMI_Q.pkl", "rb"))[0:15]
print(np.shape(X_Q))
X_Q_label = np.zeros([np.shape(X_Q)[0], 1]) + 1

X_N_test = pickle.load(open("../data/X_AMMI_N.pkl", "rb"))[1000:1330]
print(np.shape(X_N_test))
X_N_test_label = np.zeros([np.shape(X_N_test)[0], 1])

X_S_test = pickle.load(open("../data/X_AMMI_S.pkl", "rb"))[330:660]
print(np.shape(X_S_test))
X_S_test_label = np.zeros([np.shape(X_S_test)[0], 1]) + 1

X_V_test = pickle.load(open("../data/X_AMMI_V.pkl", "rb"))[330:660]
print(np.shape(X_V_test))
X_V_test_label = np.zeros([np.shape(X_V_test)[0], 1]) + 1

X_F_test = pickle.load(open("../data/X_AMMI_F.pkl", "rb"))[330:660]
print(np.shape(X_F_test))
X_F_test_label = np.zeros([np.shape(X_F_test)[0], 1]) + 1

X_Q_test = pickle.load(open("../data/X_AMMI_Q.pkl", "rb"))[15:30]
print(np.shape(X_Q_test))
X_Q_test_label = np.zeros([np.shape(X_Q_test)[0], 1]) + 1


def buildModel():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(216, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='SAME', activation='relu'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='SAME', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='SAME', activation='relu'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='SAME'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='SAME', activation='relu'),
        tf.keras.layers.LSTM(128),
        # 打平层,方便全连接层处理
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点

        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return newModel


def OutOfDatesetTest(TestData, TestLabel):
    Y_pred = model.predict(TestData)
    predict = np.argmax(Y_pred, axis=1)
    # print(predict)

    from sklearn.metrics import accuracy_score
    print("acc:")
    print(accuracy_score(TestLabel, predict))

    from sklearn.metrics import precision_score
    print("p:")
    print(precision_score(TestLabel, predict))

    from sklearn.metrics import recall_score
    print("r:")
    print(recall_score(TestLabel, predict))

    from sklearn.metrics import f1_score
    print("f1:")
    print(f1_score(TestLabel, predict))

    from sklearn.metrics import confusion_matrix  # 导入计算混淆矩阵的包

    C1 = confusion_matrix(TestLabel, predict)  # True_label 真实标签 shape=(n,1);T_predict1 预测标签 shape=(n,1)
    print(C1)
    plt.matshow(C1, cmap=plt.cm.Greens)
    plt.colorbar()
    for i in range(len(C1)):
        for j in range(len(C1)):
            plt.annotate(C1[i, j], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def train_data_without_F():
    X = np.concatenate((X_N, X_S, X_V, X_Q))
    print(np.shape(X))
    y = np.concatenate((X_N_label, X_S_label, X_V_label, X_Q_label))
    np.shape(y)
    return X, y


def train_data_without_V():
    X = np.concatenate((X_N, X_S, X_F, X_Q))
    print(np.shape(X))
    y = np.concatenate((X_N_label, X_S_label, X_F_label, X_Q_label))
    np.shape(y)
    return X, y


def train_data_without_S():
    X = np.concatenate((X_N, X_F, X_V, X_Q))
    print(np.shape(X))
    y = np.concatenate((X_N_label, X_F_label, X_V_label, X_Q_label))
    np.shape(y)
    return X, y


def train_data_without_Q():
    X = np.concatenate((X_N, X_S, X_V, X_F))
    print(np.shape(X))
    y = np.concatenate((X_N_label, X_S_label, X_V_label, X_F_label))
    np.shape(y)
    return X, y


def test_data_without_F():
    X = np.concatenate((X_N_test, X_S_test, X_V_test, X_Q_test))
    print(np.shape(X))
    y = np.concatenate((X_N_test_label, X_S_test_label, X_V_test_label, X_Q_test_label))
    np.shape(y)
    return X, y


def test_data_without_V():
    X = np.concatenate((X_N_test, X_S_test, X_F_test, X_Q_test))
    print(np.shape(X))
    y = np.concatenate((X_N_test_label, X_S_test_label, X_F_test_label, X_Q_test_label))
    np.shape(y)
    return X, y


def test_data_without_S():
    X = np.concatenate((X_N_test, X_F_test, X_V_test, X_Q_test))
    print(np.shape(X))
    y = np.concatenate((X_N_test_label, X_F_test_label, X_V_test_label, X_Q_test_label))
    np.shape(y)
    return X, y


def test_data_without_Q():
    X = np.concatenate((X_N_test, X_S_test, X_V_test, X_F_test))
    print(np.shape(X))
    y = np.concatenate((X_N_test_label, X_S_test_label, X_V_test_label, X_F_test_label))
    np.shape(y)
    return X, y


X, y = train_data_without_Q()

X_test, y_test = test_data_without_Q()

model = buildModel()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model.summary()

# 训练与验证
model.fit(X, y, epochs=50)

model.save("model/model_Q.h5")

# Y_pred = model.predict_classes(X_test)
# # 绘制混淆矩阵
# plotHeatMap(Y_test, Y_pred)

OutOfDatesetTest(X_test, y_test)
