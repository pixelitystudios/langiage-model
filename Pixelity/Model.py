from keras.models import Model
from keras.layers import LSTM, Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from collections import Counter
import nltk
import numpy as np
import os
import pandas as pd
import re
import json
import tensorflow as tf
from keras.losses import categorical_crossentropy
from keras import backend as K
import math

class LanguageModel:

    BATCH_SIZE = 32
    NUM_EPOCHS = 200
    HIDDEN_UNITS = 256
    MAX_INPUT_SEQ_LENGTH = 17
    MAX_TARGET_SEQ_LENGTH = 24
    MAX_VOCAB_SIZE = 2000
    input_counter = Counter()
    target_counter = Counter()
    input_texts = []
    target_texts = []
    input_word2idx = {}
    target_word2idx = {}
    num_encoder_tokens = 0
    num_decoder_tokens = 0
    encoder_input_data = []

    encoder_max_seq_length = 0
    decoder_max_seq_length = 0

    encoder_max_seq_length = 0
    decoder_max_seq_length = 0

    context = dict()

    model = None
    architecture = None

    train_gen = None
    test_gen = None

    path = './'

    train_num_batches = None
    test_num_batches = None

    def __init__(self, path):
        print('New model created')
        np.random.seed(42)
        self.path = path

    def store_js(self, filename, data):
        with open(filename, 'w') as f:
            f.write('module.exports = ' + json.dumps(data, indent=2))

    def compile(self, questions, answers):
        prev_words = []
        for line in questions:
            next_words = [w.lower() for w in nltk.word_tokenize(line)]
            if len(next_words) > self.MAX_TARGET_SEQ_LENGTH:
                next_words = next_words[0:self.MAX_TARGET_SEQ_LENGTH]

            if len(prev_words) > 0:
                self.input_texts.append(prev_words)
                for w in prev_words:
                    self.input_counter[w] += 1

            prev_words = next_words

        prev_words = []
        for line in answers:
            next_words = [w.lower() for w in nltk.word_tokenize(line)]
            if len(next_words) > self.MAX_TARGET_SEQ_LENGTH:
                next_words = next_words[0:self.MAX_TARGET_SEQ_LENGTH]

            if len(prev_words) > 0:
                target_words = next_words[:]
                target_words.insert(0, '<SOS>')
                target_words.append('<EOS>')
                for w in target_words:
                    self.target_counter[w] += 1
                self.target_texts.append(target_words)

            prev_words = next_words
        for idx, word in enumerate(self.input_counter.most_common(self.MAX_VOCAB_SIZE)):
            self.input_word2idx[word[0]] = idx + 2
        for idx, word in enumerate(self.target_counter.most_common(self.MAX_VOCAB_SIZE)):
            self.target_word2idx[word[0]] = idx + 1

        self.input_word2idx['<PAD>'] = 0
        self.input_word2idx['<UNK>'] = 1
        self.target_word2idx['<UNK>'] = 0

        self.input_idx2word = dict([(idx, word) for word, idx in self.input_word2idx.items()])

        self.target_idx2word = dict([(idx, word) for word, idx in self.target_word2idx.items()])

        self.num_encoder_tokens = len(self.input_idx2word)
        self.num_decoder_tokens = len(self.target_idx2word)


        for input_words, target_words in zip(self.input_texts, self.target_texts):
            encoder_input_wids = []
            for w in input_words:
                w2idx = 1  # default [UNK]
                if w in self.input_word2idx:
                    w2idx = self.input_word2idx[w]
                encoder_input_wids.append(w2idx)

            self.encoder_input_data.append(encoder_input_wids)
            self.encoder_max_seq_length = max(len(encoder_input_wids), self.encoder_max_seq_length)
            self.decoder_max_seq_length = max(len(target_words), self.decoder_max_seq_length)


        self.context['num_encoder_tokens'] = self.num_encoder_tokens
        self.context['num_decoder_tokens'] = self.num_decoder_tokens
        self.context['encoder_max_seq_length'] = self.encoder_max_seq_length
        self.context['decoder_max_seq_length'] = self.decoder_max_seq_length

        self.encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        self.encoder_embedding = Embedding(input_dim=self.num_encoder_tokens, output_dim=self.HIDDEN_UNITS,
                                      input_length=self.encoder_max_seq_length, name='encoder_embedding')
        self.encoder_lstm = LSTM(units=self.HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        self.encoder_outputs, self.encoder_state_h, self.encoder_state_c = self.encoder_lstm(self.encoder_embedding(self.encoder_inputs))
        self.encoder_states = [self.encoder_state_h, self.encoder_state_c]

        self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens), name='decoder_inputs')
        self.decoder_lstm = LSTM(units=self.HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        self.decoder_outputs, self.decoder_state_h, self.decoder_state_c = self.decoder_lstm(self.decoder_inputs,
                                                                         initial_state=self.encoder_states)
        self.decoder_dense = Dense(units=self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        self.model = Model([self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)
        optimizer = Adam(lr=0.005)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[self.ppx])
        self.architecture = self.model.to_json()
        X_train, X_test, y_train, y_test = train_test_split(self.encoder_input_data, self.target_texts, test_size=0.05,random_state=42)
        self.train_gen = self.generate_batch(X_train, y_train)
        self.test_gen = self.generate_batch(X_test, y_test)

        self.train_num_batches = len(X_train) // self.BATCH_SIZE
        self.test_num_batches = len(X_test) // self.BATCH_SIZE

    def fit(self, epochs):
        self.NUM_EPOCHS = epochs
        checkpoint = ModelCheckpoint(filepath=os.path.join(self.path,'model/word-weights.h5'), save_best_only=True)
        self.model.fit_generator(generator=self.train_gen, steps_per_epoch=self.train_num_batches,epochs=self.NUM_EPOCHS,verbose=1, validation_data=self.test_gen, validation_steps=self.test_num_batches,callbacks=[checkpoint])

        encoder_model = Model(self.encoder_inputs, self.encoder_states)
        encoder_model.save(os.path.join(self.path,'model/encoder-weights.h5'))

        new_decoder_inputs = Input(batch_shape=(1, None, self.num_decoder_tokens), name='new_decoder_inputs')
        new_decoder_lstm = LSTM(units=self.HIDDEN_UNITS, return_state=True, return_sequences=True, name='new_decoder_lstm',
                                stateful=True)
        new_decoder_outputs, _, _ = new_decoder_lstm(new_decoder_inputs)
        new_decoder_dense = Dense(units=self.num_decoder_tokens, activation='softmax', name='new_decoder_dense')
        new_decoder_outputs = new_decoder_dense(new_decoder_outputs)
        new_decoder_lstm.set_weights(self.decoder_lstm.get_weights())
        new_decoder_dense.set_weights(self.decoder_dense.get_weights())

        new_decoder_model = Model(new_decoder_inputs, new_decoder_outputs)

        new_decoder_model.save(os.path.join(self.path,'model/decoder-weights.h5'))


    def ppx(self, y_true, y_pred):
        loss = categorical_crossentropy(y_true, y_pred)
        perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
        return perplexity

    def generate_batch(self, input_data, output_text_data):
        num_batches = len(input_data) // self.BATCH_SIZE
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx *  self.BATCH_SIZE
                end = (batchIdx + 1) *  self.BATCH_SIZE
                encoder_input_data_batch = pad_sequences(input_data[start:end],  self.encoder_max_seq_length)
                decoder_target_data_batch = np.zeros(shape=( self.BATCH_SIZE,  self.decoder_max_seq_length,  self.num_decoder_tokens))
                decoder_input_data_batch = np.zeros(shape=( self.BATCH_SIZE,  self.decoder_max_seq_length,  self.num_decoder_tokens))
                for lineIdx, target_words in enumerate(output_text_data[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in  self.target_word2idx:
                            w2idx =  self.target_word2idx[w]
                        decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                        if idx > 0:
                            decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    def save_model(self):
        np.save(os.path.join(self.path,'model/vectors/word-input-word2idx.npy'), self.input_word2idx)
        np.save(os.path.join(self.path,'model/vectors/word-input-idx2word.npy'), self.input_idx2word)
        np.save(os.path.join(self.path,'model/vectors/word-target-word2idx.npy'), self.target_word2idx)
        np.save(os.path.join(self.path,'model/vectors/word-target-idx2word.npy'), self.target_idx2word)

        # Store necessary mappings for tfjs
        self.store_js(os.path.join(self.path,'model/mappings/input-word2idx.js'), self.input_word2idx)
        self.store_js(os.path.join(self.path,'model/mappings/input-idx2word.js'), self.input_idx2word)
        self.store_js(os.path.join(self.path,'model/mappings/target-word2idx.js'), self.target_word2idx)
        self.store_js(os.path.join(self.path,'model/mappings/target-idx2word.js'), self.target_idx2word)

        np.save(os.path.join(self.path,'model/vectors/word-context.npy'), self.context)
        self.store_js(os.path.join(self.path,'model/mappings/word-context.js'), self.context)
        open(os.path.join(self.path,'model/vectors/word-architecture.json'), 'w').write(self.architecture)