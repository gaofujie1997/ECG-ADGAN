import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import InputSpec, Layer, Dropout, MaxPooling1D, Concatenate, Input, Dense, Reshape, \
    Flatten, Activation, UpSampling1D, Conv1D, Bidirectional, LSTM, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints


class MinibatchDiscrimination(Layer):
    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
                                 initializer=self.init,
                                 name='kernel',
                                 regularizer=self.W_regularizer,
                                 trainable=True,
                                 constraint=self.W_constraint)

        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1] + self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': 'GlorotUniform',
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DCGAN:
    def __init__(self, input_shape=(216, 1), latent_size=100, random_sine=True, scale=1, minibatch=False):
        self.input_shape = input_shape
        self.latent_size = latent_size
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.optimizer = optimizer
        self.random_sine = random_sine
        self.scale = scale
        self.minibatch = minibatch
        self.discrimintor = self.bulid_discrimintor()
        self.discrimintor.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_size,))
        signal = self.generator(z)
        self.discrimintor.trainable = False
        valid = self.discrimintor(signal)
        self.combine = Model(z, valid)
        self.combine.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        model = Sequential(name='Generator_v1')
        model.add(Reshape((self.latent_size, 1)))
        model.add(Bidirectional(LSTM(16, return_sequences=True)))

        model.add(Conv1D(32, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling1D())
        model.add(Conv1D(16, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling1D())
        model.add(Conv1D(8, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv1D(1, kernel_size=8, padding="same"))
        model.add(Flatten())

        model.add(Dense(self.input_shape[0]))
        model.add(Activation('tanh'))
        model.add(Reshape(self.input_shape))
        noise = Input(shape=(self.latent_size,))
        signal = model(noise)
        model.summary()

        return Model(inputs=noise, outputs=signal)

    def bulid_discrimintor(self):
        signal = Input(shape=self.input_shape)

        flat = Flatten()(signal)
        mini_disc = MinibatchDiscrimination(10, 3)(flat)

        md = Conv1D(8, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(signal)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3)(md)

        md = Conv1D(16, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(md)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3, strides=2)(md)

        md = Conv1D(32, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3, strides=2)(md)

        md = Conv1D(64, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
        md = LeakyReLU(alpha=0.2)(md)
        md = Dropout(0.25)(md)
        md = MaxPooling1D(3, strides=2)(md)
        md = Flatten()(md)
        concat = Concatenate()([md, mini_disc])
        validity = Dense(1, activation='sigmoid')(concat)

        return Model(inputs=signal, outputs=validity, name="Discriminator")

    def train(self, epochs, X_train, batch_size=128, save_interval=50, save=False, save_model_interval=100):
        vaild = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        progress = {'D_loss': [],
                    'G_loss': [],
                    'acc': []}
        flag = 0
        for epoch in range(epochs):

            # -------------------
            # Train discriminator
            # -------------------

            # select a random batch of signals
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            signals = X_train[idx]

            # sample noise and generatir a batch of new signals
            noise = self.generate_noise(batch_size, self.random_sine)
            gen_signals = self.generator.predict(noise)

            # train the discriminator (real signals labeled as 1 and fake labeled as 0)
            d_loss_real = self.discrimintor.train_on_batch(signals, vaild)
            d_loss_fake = self.discrimintor.train_on_batch(gen_signals, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -------------------
            # Train Generator
            # -------------------
            stop = 1000

            if flag == 0:
                if epoch > stop and (100 * d_loss[1]) > 49.5 and (100 * d_loss[1] < 50.5):
                    self.generator.trainable = False
                    flag = 1
                else:
                    g_loss = self.combine.train_on_batch(noise, vaild)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            progress['D_loss'].append((d_loss[0]))
            progress['acc'].append(d_loss[1])
            progress['G_loss'].append((g_loss))

            if epoch % save_interval == 0:
                self.save_image(epoch)
                self.loss_plot(progress, path="loss/%d.png" % epoch)
            if save and epoch > stop:
                if os.path.isdir('model/') != True:
                    os.mkdir('model/')
                if (epoch % save_model_interval == 0 and epoch > 0):
                    self.discrimintor.save('model/%d.h5' % epoch)

        self.loss_plot(progress, path="loss/%d.png" % epoch)
        self.save_image(epoch)
        self.discrimintor.save('model/%d.h5' % epoch)

    def save_image(self, epoch):
        if os.path.isdir('ecg_image/') != True:
            os.mkdir('ecg_image/')
        r, c = 2, 2
        noise = self.generate_noise(r * c, self.random_sine)
        signals = self.generator.predict(noise) * self.scale
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].plot(signals[cnt])
                cnt += 1
        fig.savefig('ecg_image/%d.png' % epoch)
        plt.close()

    def loss_plot(self, hist, path):
        x = range(len(hist['D_loss']))

        y1 = hist['D_loss']
        y2 = hist['G_loss']

        plt.plot(x, y1, label='D_loss')
        plt.plot(x, y2, label='G_loss')

        plt.xlabel('epoch')
        plt.ylabel('loss')

        plt.legend(loc=1)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(path)

        plt.close()

    def generate_noise(self, batch_size, sinwave=False):
        if sinwave:
            x = np.linspace(-np.pi, np.pi, self.latent_size)
            noise = 0.1 * np.random.random_sample((batch_size, self.latent_size)) + 0.9 * np.sin(x)
        else:
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_size))
        return noise

    def specify_range(self, signals, min_val=-1, max_val=1):
        if signals is None:
            raise ValueError("No signals data.")
        if type(signals) != np.ndarray:
            signals = np.array(signals)
        select_signals = []
        for sg in signals:
            min_sg = np.min(sg)
            max_sg = np.max(sg)

            if (min_sg >= min_val and max_sg <= max_val):
                select_signals.append(sg)

        return np.array(select_signals)
