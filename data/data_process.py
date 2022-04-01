import pickle
import pywt
import wfdb
import numpy as np


def denoise(data):
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def getDataSet(number, X_data, Y_data, ecgClassSet):
    print("Reading:No." + number + " ecg...")
    record = wfdb.rdrecord('ecg_data/' + number, channel_names=['V6'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)

    annotation = wfdb.rdann('ecg_data/' + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol

    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end

    while i < j:
        try:
            lable = ecgClassSet.index(Rclass[i])
            x_train = rdata[Rlocation[i] - 80:Rlocation[i] + 136]
            X_data.append(x_train)
            Y_data.append(lable)
            i += 1
        except ValueError:
            i += 1
    return


def loadData(numberSet, ecgClassSet, pklname):
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet, ecgClassSet)

    dataSet = np.array(dataSet).reshape(-1, 216)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)

    X = train_ds[:, :216].reshape(-1, 216, 1)

    with open(pklname, 'wb') as f:
        pickle.dump(X, f)


numberSet_Train = ['101', '103', '112', '113', '115', '117', '121', '122', '123', '230']

numberSet_Test = ['100', '102', '104', '105', '106', '107', '108', '109', '111', '114', '116', '118', '119', '124',
                  '200',
                  '201', '202', '203', '205', '207', '208', '209', '210', '212', '213', '214', '215', '217', '219',
                  '220',
                  '221', '222', '223', '228', '231', '232', '233', '234']

ecgClassSet_N = ['N', 'L', 'R']
ecgClassSet_AN = ['/', 'f', 'Q', 'F', 'V', 'E', 'A', 'a', 'J', 'S', 'j', 'e']

pklname = '../data/N.pkl'

loadData(numberSet_Train, ecgClassSet_N, pklname)
