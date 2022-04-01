from tensorflow.keras.models import load_model
import pickle
from ablation_model import MinibatchDiscrimination
from uilts.utils import specify_range


def changeshape(path):
    X_train = pickle.load(open(path, "rb"))
    X_train = specify_range(X_train, -2, 2) / 2
    X_train = X_train.reshape(-1, 216, 1)
    return X_train


def compute(N_path, UN_path, model_path):
    X_train_N = changeshape(N_path)
    X_train_UN = changeshape(UN_path)

    model_d = load_model(model_path, custom_objects={'MinibatchDiscrimination': MinibatchDiscrimination})

    critic_N = model_d.predict(X_train_N)
    critic_UN = model_d.predict(X_train_UN)

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i, decision in enumerate(critic_N):
        if decision > 0.5:
            TP += 1
        else:
            FP += 1

    for i, decision in enumerate(critic_UN):
        if decision < 0.5:
            TN += 1
        else:
            FN += 1

    print(TP, TN, FP, FN)

    acc = (TP + TN) / (TP + TN + FP + FN)

    precision = (TP + 0.000001) / (TP + FP + 0.000001)

    recall = (TP + 0.000001) / (TP + FN + 0.000001)

    F1 = 2 * (precision * recall) / (precision + recall)

    print("acc=%.4f" % acc, " precision=%.4f" % precision, " recall=%.4f" % recall, " F1=%.4f" % F1)

    return acc, precision, recall, F1


def metric():
    model_num = 'a3'
    N_path = "../data/Test_N.pkl"
    UN_path = "../data/Test_AN.pkl"
    compute(N_path, UN_path, 'ablation_model/saved_models/%s.h5' % model_num)


if __name__ == "__main__":
    metric()
