# Claims & Vehicle Fault-based Diagnostic Action Prediction (CVFDA) Model
# This component uses a vehicle's fault and claim history to predict the most suitable diagnostic actions to
# address a particular fault. This prediction doesn't explicitly consider the sequence in which the actions
# should be executed.
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Input, Flatten, Dense, Layer, Dropout, BatchNormalization, MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import plot_model, register_keras_serializable
from keras.metrics import AUC, Precision, Recall
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import LabelEncoder

import data_preprocessing

# Constants
MODEL_NAME = 'cvf_da'
CATEGORICAL_FEATURES = ['model', 'modelyear', 'driver', 'plant', 'engine', 'transmission', 'module', 'dtcbase', 'faulttype',
                        'dtcfull', 'year', 'month', 'dayOfWeek', 'weekOfYear', 'season', 'i_original_vfg_code',
                        'softwarepartnumber', 'hardwarepartnumber', 'i_p_css_code', 'i_original_ccc_code',
                        'i_original_function_code', 'i_original_vrt_code', 'i_current_vfg_code', 'i_current_function_code',
                        'i_current_vrt_code',	'i_cpsc_code', 'i_cpsc_vfg_code', 'i_css_code', 'v_transmission_code',
                        'v_drive_code', 'v_engine_code', 'ic_repair_dealer_id', 'ic_eng_part_number', 'ic_serv_part_number',
                        'ic_part_suffix', 'ic_part_base', 'ic_part_prefix', 'ic_causal_part_id', 'ic_repair_country_code']
NUMERICAL_FEATURES = ['elapsedTimeSec', 'timeSinceLastActivitySec', 'odomiles', 'vehicleAgeAtSession',
                      'daysSinceWarrantyStart', 'i_mileage', 'i_time_in_service', 'i_months_in_service']
BATCH_SIZE = 128
EPOCHS = 2


# Convolution with Cross Convolutional Filters
@register_keras_serializable('models')
class CrossConv2D(Layer):
    def __init__(self, filters, kernel_size, **kwargs):
        super(CrossConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv1 = tf.keras.layers.Conv2D(filters, (kernel_size[0], 1), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, (1, kernel_size[1]), activation='relu', padding='same')

    def call(self, inputs):
        conv1_output = self.conv1(inputs)
        conv2_output = self.conv2(inputs)
        return conv1_output + conv2_output

    def get_config(self):
        base_config = super().get_config()
        config = {"filters": self.filters, "kernel_size": self.kernel_size}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def load_data(path):
    data_df = data_preprocessing.load_data(path)
    data_df = data_df.iloc[:, 1:]  # remove index column
    return data_df


def encode_categorical_features(data_df, label_encoder):
    data_df.loc[:, 'otxsequence'] = label_encoder.fit_transform(data_df['otxsequence'])

    for feature in CATEGORICAL_FEATURES:
        data_df[feature] = data_df[feature].astype('str').astype('category')
        data_df[feature] = label_encoder.fit_transform(data_df[feature])
    return data_df


def variable_embedding(data_df):
    input_layers = []
    embedding_layers = []

    for col in CATEGORICAL_FEATURES:
        num_unique_categories = data_df[col].nunique()

        # Create input layer for each category
        input_layer = Input(shape=(1,), name=f"{col}_input")
        input_layers.append(input_layer)

        # Create embedding layer for each category
        embedding = Embedding(num_unique_categories, 50, input_length=1, name=f"{col}_embedding")(input_layer)

        # Flatten the embedding layer
        embedding_flatten = Flatten()(embedding)
        embedding_layers.append(embedding_flatten)

    # Define the input layer for the numerical features
    num_input = Input(shape=(len(NUMERICAL_FEATURES),), name='numerical_input')

    # Pass the numerical inputs through a MLP
    hidden1 = Dense(128, activation='relu')(num_input)
    hidden2 = Dense(64, activation='relu')(hidden1)
    num_output = Dense(32, activation='relu')(hidden2)  # This is the embedding of the continuous features

    # Append num_output to the list of embedding layers
    embedding_layers.append(num_output)

    return input_layers, num_input, embedding_layers


def reshaping_inputs(embeddings_concat):
    print(f'Original embeddings shape: {embeddings_concat.shape}')
    total_features = embeddings_concat.shape[1] # the total number of features after concatenation
    sqrt_features = int(np.ceil(np.sqrt(total_features))) # the nearest square number greater than total_features

    flattened = tf.keras.layers.Flatten()(embeddings_concat)
    dense_for_reshape = tf.keras.layers.Dense(sqrt_features*sqrt_features, activation='relu')(flattened)

    embeddings_reshaped = tf.keras.layers.Reshape((sqrt_features, sqrt_features, 1))(dense_for_reshape)
    print(f'Reshaped embeddings shape: {embeddings_reshaped.shape}')
    return embeddings_reshaped


def compile_model(input_layers, num_input, output_layer):
    model = Model(inputs=input_layers + [num_input], outputs=[output_layer])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', AUC(), Precision(),
                                                                              Recall()])
    plot_model(model, to_file=f'out/models/cvf_da/{MODEL_NAME}_layers.png', show_shapes=True, show_layer_names=True)
    return model


def prepare_train_test_data(data_df):
    target = to_categorical(data_df['otxsequence'])
    features = data_df.drop(columns='otxsequence')
    input_train, input_test, output_train, output_test = train_test_split(features,
                                                                          target,
                                                                          test_size=0.2,
                                                                          random_state=42,
                                                                          stratify=target)
    train_input = [input_train[feature].values for feature in CATEGORICAL_FEATURES] +\
                  [input_train[NUMERICAL_FEATURES].values]
    test_input = [input_test[feature].values for feature in CATEGORICAL_FEATURES] + \
                 [input_test[NUMERICAL_FEATURES].values]

    return train_input, test_input, output_train, output_test


def train_model(model, train_input, y_train, test_input, y_test):
    early_stopping_monitor = EarlyStopping(
        monitor='accuracy',
        min_delta=0.001,  # minimum change to qualify as an improvement
        patience=5,  # number of epochs with no improvement after which training will be stopped
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )

    history = model.fit(
        train_input,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(test_input, y_test),
        callbacks=[early_stopping_monitor]
    )

    return history
