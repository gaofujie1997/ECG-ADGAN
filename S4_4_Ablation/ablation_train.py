import pickle
from ablation_model import DCGAN

X_train = pickle.load(open("../data/Train.pkl", "rb"))
EPOCHS = 20000
LATENT_SIZE = 100
SAVE_INTRIVAL = 100
SAVE_MODEL_INTERVAL = 20
BATCH_SIZE = 128
INPUT_SHAPE = (216, 1)
RANDOM_SINE = False
SCALE = 2
MINIBATCH = True
SAVE_MODEL = True
SAVE_REPORT = True
dcgan = DCGAN(INPUT_SHAPE, LATENT_SIZE, random_sine=RANDOM_SINE, scale=SCALE, minibatch=MINIBATCH)
X_train = dcgan.specify_range(X_train, -2, 2) / 2
X_train = X_train.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
dcgan.train(EPOCHS, X_train, BATCH_SIZE, SAVE_INTRIVAL, save=SAVE_MODEL, save_model_interval=SAVE_MODEL_INTERVAL)
