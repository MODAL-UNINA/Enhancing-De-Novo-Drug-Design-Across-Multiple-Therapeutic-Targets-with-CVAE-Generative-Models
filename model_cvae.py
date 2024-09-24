#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %%
import numpy as np
import tensorflow as tf 
import tensorflow_addons as tfa
import time
from tqdm.auto import tqdm
import matplotlib.pyplot as plt   
from utils import *

class Encoder(tf.keras.Model):
    def __init__(self, batch_size, unit_size, latent_size, vocab_size, n_rnn_layer, num_prop, seq_length, embedding):
        super().__init__()
        self.batch_size = batch_size
        self.unit_size = unit_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.n_rnn_layer = n_rnn_layer
        self.num_prop = num_prop
        self.seq_length = seq_length
        self.embedding = embedding

        initializer= tf.keras.initializers.glorot_uniform()
        ker_reg = tf.keras.regularizers.l2(0.00001)

        self.rnn_layer_encode = [tf.keras.layers.LSTM(self.unit_size, kernel_initializer=initializer,
                                            return_sequences=True, return_state=True, kernel_regularizer=ker_reg) for _ in range(self.n_rnn_layer)]

        self.dense_mean = tf.keras.layers.Dense(self.latent_size, name='mean', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                                kernel_regularizer=ker_reg)
        self.dense_logvar = tf.keras.layers.Dense(self.latent_size, name='logvar', kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                                    kernel_regularizer=ker_reg)
        
    
    def call(self, inputs_XCL, training=False):
        X, C, L = inputs_XCL
        X = self.embedding(X, training=training)
        C = tf.tile(tf.expand_dims(C, 1), [1, self.seq_length, 1])
        inputs = tf.concat([X, C], axis=-1)

        mask = tf.sequence_mask(L, maxlen=self.seq_length, dtype=tf.bool)

        output_state = []
        for i in range(self.n_rnn_layer):
            inputs, *state_out = self.rnn_layer_encode[i](inputs, training = training, initial_state = None, mask=mask)
            output_state.append(state_out)

        c, h = output_state[-1]

        z_mean = self.dense_mean(h, training=training)
        z_logvar = self.dense_logvar(h, training=training)

        epsilon =  tf.random.normal(shape=(tf.shape(z_mean)[0], self.latent_size), mean=0.0, stddev=1.0)
        z = z_mean + tf.exp(0.5 * z_logvar) * epsilon

        return z, z_mean, z_logvar



class Decoder(tf.keras.Model):
    def __init__(self, batch_size, unit_size, latent_size, vocab_size, n_rnn_layer, num_prop, embedding):
        super().__init__()
        self.batch_size = batch_size
        self.unit_size = unit_size
        self.latent_size = latent_size
        self.vocab_size = vocab_size
        self.n_rnn_layer = n_rnn_layer
        self.num_prop = num_prop
        self.embedding = embedding

        initializer= tf.keras.initializers.glorot_uniform()
        ker_reg = tf.keras.regularizers.l2(0.00001)

        self.rnn_layer_decode = [tf.keras.layers.LSTM(self.unit_size, kernel_initializer=initializer,
                                    return_sequences=True, return_state=True, kernel_regularizer=ker_reg) for _ in range(self.n_rnn_layer)]

        self.dense = tf.keras.layers.Dense(self.vocab_size, kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1),
                                        kernel_regularizer=ker_reg)
        self.softmax = tf.keras.layers.Softmax()

    
    def call(self, inputs_XZCL, state=None, training=False):

        X, Z, C, L = inputs_XZCL       
        seq_length = tf.shape(X)[1]
        Z = tf.tile(tf.expand_dims(Z, axis=1), [1, seq_length, 1])
        C = tf.expand_dims(C, 1)
        C = tf.cast(tf.tile(C, [1, seq_length, 1]), tf.float32)
        X = self.embedding(X, training=training)

        inputs = tf.concat([Z, X, C], axis=-1)
        mask = tf.sequence_mask(L, maxlen=seq_length, dtype=tf.bool)


        if training:
            for i in range(self.n_rnn_layer):
                inputs, *output_state = self.rnn_layer_decode[i](inputs, training=training, initial_state = None, mask=mask)
            X2 = inputs

        elif not training:
            if state is None:
                state = [[tf.zeros([inputs_XZCL[0].shape[0], self.unit_size], tf.float32), 
                        tf.zeros([inputs_XZCL[0].shape[0], self.unit_size], tf.float32)] 
                        for _ in range(self.n_rnn_layer)]
            output_state = []
            for i in range(self.n_rnn_layer):
                inputs, *state_out =  self.rnn_layer_decode[i](inputs, training=training, initial_state=state[i], mask=mask)
                output_state.append(state_out)    
            X2 = inputs

        X2 = tf.reshape(X2, [-1, tf.shape(X2)[-1]])
        X3_logits = self.dense(X2, training=training)
        X3_logits = tf.reshape(X3_logits, [-1, seq_length, tf.shape(X3_logits)[-1]])
        X3 = self.softmax(X3_logits)
        return X3, X3_logits, output_state


class Model(tf.keras.Model):
    def __init__(self,vocab):
        super(Model, self).__init__()
        self.batch_size = 100 #2048 initial; close to 100 for tuning
        self.unit_size = 512
        self.latent_size = 200 #200
        self.vocab_size = len(vocab)
        self.n_rnn_layer = 3
        self.num_prop = 6
        self.seq_length = 120
        self.lr = 1e-5 #1e-4 initial #1e-5 for tuning
        self.kl_beta = 1 # 1
        self.vocab = vocab
        self.range_list = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]


        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.embedding = tf.keras.layers.Embedding(self.latent_size,self.vocab_size,  embeddings_initializer=
                                                    tf.keras.initializers.random_uniform(minval=-0.1, maxval=0.1))
        self.encoder = Encoder(self.batch_size, self.unit_size, self.latent_size, self.vocab_size, self.n_rnn_layer, self.num_prop, self.seq_length,
                                self.embedding)
        self.decoder = Decoder(self.batch_size, self.unit_size, self.latent_size, self.vocab_size, self.n_rnn_layer, self.num_prop,
                                self.embedding)

    @tf.function
    def call(self, inputs, state=None, training=False):
        X, C, L = inputs
        z, z_mean, z_logvar = self.encoder([X, C, L], training=training)
        Y_hat, Y_hat_logits, state = self.decoder([X, z, C, L], state=state, training=training)
        return Y_hat, Y_hat_logits, z_mean, z_logvar, z, state
    
    def cal_latent_loss(self, mean, log_sigma):
        latent_loss = tf.reduce_mean(-0.5*(1+log_sigma-tf.square(mean)-tf.exp(log_sigma)))
        return latent_loss
    
    def loss(self, Y, L, Y_hat, Y_hat_logits, z_mean, z_log_var):
        weights = tf.sequence_mask(L, maxlen=self.seq_length, dtype=tf.float32)
        reconstruction_loss = tf.reduce_mean(tfa.seq2seq.sequence_loss(logits=Y_hat_logits, targets=Y, weights=weights))
        kl_loss = self.cal_latent_loss(z_mean, z_log_var)
        total_loss = reconstruction_loss + self.kl_beta*kl_loss
        
        return {'loss': total_loss, 'recostruction': reconstruction_loss, 'KL': kl_loss}
    

    @tf.function
    def train_batch(self, data):
        X, C, Y, L = data

        with tf.GradientTape() as tape:
            Y_hat, Y_hat_logits, z_mean, z_log_var, z, state = self.call([X, C, L ], training=True)
            loss_batch = self.loss(Y, L, Y_hat, Y_hat_logits, z_mean, z_log_var)
            loss_batch['Regularization'] = tf.reduce_mean(self.encoder.losses) + tf.reduce_mean(self.decoder.losses)
            loss_batch['loss'] += loss_batch['Regularization']
        gradients = tape.gradient(loss_batch['loss'], self.encoder.trainable_variables+self.decoder.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.encoder.trainable_variables+self.decoder.trainable_variables))
        return loss_batch
    
    def train(self, dataset_train, dataset_val, epochs, patience):
        history = {}
        best_val_loss = tf.constant(np.inf)
        count = 0
        X_train, Y_train, C_train, L_train = dataset_train
        X_val, Y_val, C_val, L_val = dataset_val


        from tqdm.auto import tqdm
        for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
            start = time.time()
            loss_batches = {}

            for i in range(len(X_train)// self.batch_size):
                X_batch = X_train[i*self.batch_size:i*self.batch_size + self.batch_size]
                Y_batch = Y_train[i*self.batch_size:i*self.batch_size + self.batch_size]
                C_batch = C_train[i*self.batch_size:i*self.batch_size + self.batch_size]
                L_batch = L_train[i*self.batch_size:i*self.batch_size + self.batch_size]

                loss_batch = self.train_batch([X_batch, C_batch, Y_batch, L_batch])
                for key, valu in loss_batch.items():
                    loss_batches.setdefault(key, []).append(valu)
                
            for i in range(len(X_val)//self.batch_size):
                X_val_batch = X_val[i*self.batch_size:i*self.batch_size + self.batch_size]
                Y_val_batch = Y_val[i*self.batch_size:i*self.batch_size + self.batch_size]
                C_val_batch = C_val[i*self.batch_size:i*self.batch_size + self.batch_size]
                L_val_batch = L_val[i*self.batch_size:i*self.batch_size + self.batch_size]
                
                Y_hat_val, Y_hat_val_batch, z_mean_val_batch, z_log_var_val_batch, z_val_batch, state_val_batch = self.call(
                    [X_val_batch, C_val_batch, L_val_batch], training=True)# add training true
                
                loss_val_batch = self.loss(Y_val_batch, L_val_batch, Y_hat_val, Y_hat_val_batch, z_mean_val_batch, z_log_var_val_batch
                                            )
                
                for key, valu in loss_val_batch.items():
                    loss_batches.setdefault(key+'_val', []).append(valu)

            for key, valu in loss_batches.items():
                history.setdefault(key, []).append(tf.reduce_mean(valu))

            stampa = f'Epoch: {epoch + 1}/{epochs}, '
            for key, valu in history.items():
                stampa += f'{key}: {valu[-1].numpy():.4f}, '

            stampa += f'Time: {time.time() - start:.2f}s'
            print(stampa)
            
            if history['loss_val'][-1] < best_val_loss:
                best_val_loss = history['loss_val'][-1]
                count = 0
            else:
                count += 1

            #Early stopping
            if count == patience:
                print('Early stopping')
                return history

        return history

    def generate(self, C, start_codon):
        z = tf.keras.backend.random_normal(shape=(C.shape[0], self.latent_size), mean=0.0, stddev=1.0)
        C = tf.cast(C, tf.float32)
        X_pred = tf.cast(start_codon, tf.int32)
        preds = []
        state = None
        L = np.ones(shape=(C.shape[0],), dtype=np.int32)
        for i in range(self.seq_length):

            if i == 0:
                Y_hat, _, state = self.decoder([X_pred, z, C, L], state=None, training=False)
                Y_hat = tf.argmax(Y_hat, axis=-1)
            else:
                Y_hat, _, state = self.decoder([Y_hat, z, C, L], state=state, training=False)
                Y_hat = tf.argmax(Y_hat, axis=-1)

            preds.append(Y_hat)
        
        preds = tf.stack(preds, axis=1)
        preds = tf.squeeze(preds, axis=2)
        preds = tf.cast(preds, tf.int32)
        return preds
    
    def plot_loss(self, history, path):
        plt.figure(figsize=(10, 5))
        plt.plot(history['loss'], label='Train')
        plt.plot(history['loss_val'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    def plot_loss_kl(self, history, path):
        plt.figure(figsize=(10, 5))
        plt.plot(history['KL'], label='Train')
        plt.plot(history['KL_val'], label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('KL Loss')
        plt.legend()
        plt.show()



def load_model(args):
    model = Model(args.vocab)
    dummy_data = (tf.zeros((1, args.seq_length), dtype=tf.int32),
                                tf.zeros((1, args.num_properties), dtype=tf.float32),
                                tf.constant([args.seq_length]))
    _ = model(dummy_data)

    model.load_weights(str(args.save_dir) + f'/model_{args.grammar}_{args.model_type}_{args.target}_weights.h5')

    return model
# %%
