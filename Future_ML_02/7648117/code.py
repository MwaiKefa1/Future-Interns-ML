import os
import logging
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Type, List, Tuple
import seaborn as sns
from sklearn import metrics
import requests

class CategoryPredictor():
    __type__ = 'classification'
    def __init__(self):
        pass

    def train(self, X_train, y_train):
        os.system('mkdir .classifier_temp')
        maxWords = 500
        batch_size = 32
        epochs = 100
        drop_ratio = 0.3

        # Pre process
        tokenizer = keras.preprocessing.text.Tokenizer(num_words=maxWords, char_level=False)
        tokenizer.fit_on_texts(X_train)
        x_train = tokenizer.texts_to_matrix(X_train.values)

        # Post processing
        encoder = LabelEncoder()
        encoder.fit(y_train)
        y_train = encoder.transform(y_train)

        num_classes = np.max(y_train) + 1
        y_train = keras.utils.to_categorical(y_train, num_classes)

        # Build the model
        model = Sequential()

        model.add(Dense(60,activation='relu', input_shape=(maxWords,)))
        model.add(Dense(40,activation='relu'))
        model.add(Dropout(drop_ratio))
        model.add(Dense(30,activation='relu'))
        model.add(Dropout(drop_ratio))
        model.add(Dense(20,activation='relu'))
        model.add(Dropout(drop_ratio/2))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        path_nn = '.classifier_temp/model.h5'
        callback_save = keras.callbacks.ModelCheckpoint(path_nn, save_best_only=True, monitor='accuracy', mode='max')
        history_closed = model.fit(
            x_train, 
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_split=0.1,
            callbacks=[callback_save],
        )
        plot_history(history_closed, figsize=(10, 5), metric='accuracy')

        self.model = keras.models.load_model(path_nn)
        self.tokenizer = tokenizer
        self.encoder = encoder
        os.system('rm -r .classifier_temp')

    def save(self, base_path:str, version:str='latest'):
        path_nn = f'{base_path}/v{version}_model.h5'
        path_encoder = f'{base_path}/v{version}_encoder.pickle'
        path_tokenizer = f'{base_path}/v{version}_tokenizer.pickle'

        with open(path_tokenizer, 'wb') as file:
            pickle.dump(self.tokenizer, file)
        with open(path_encoder, 'wb') as file:
            pickle.dump(self.encoder, file)
        self.model.save(path_nn)

    @classmethod
    def load_model(cls, base_path:str, version:str='latest'):
        c = CategoryPredictor()
        path_nn = f'{base_path}/v{version}_model.h5'
        path_encoder = f'{base_path}/v{version}_encoder.pickle'
        path_tokenizer = f'{base_path}/v{version}_tokenizer.pickle'

        with open(path_encoder, 'rb') as file:
            c.encoder = pickle.load(file)
        with open(path_tokenizer, 'rb') as file:
            c.tokenizer = pickle.load(file)
        c.model = keras.models.load_model(path_nn)

        logging.info('[ML] CategoryPredictor init done')
        return c

    def predict(self, X):
        _preprocessed: np.array = self.tokenizer.texts_to_matrix(X)
        _predictions: np.array = self.model.predict(_preprocessed)
        _postprocessed = [self.encoder.classes_[np.argmax(_prediction)] for _prediction in _predictions]
        return _postprocessed

    def predict_one(self, description:str) -> str:
        _preprocessed: np.array = self.tokenizer.texts_to_matrix(np.array([description]))
        _prediction: np.array = self.model.predict(_preprocessed)
        _postprocessed: str = self.encoder.classes_[np.argmax(_prediction)]
        return _postprocessed

    def shutdown(self) -> bool:
        return True
