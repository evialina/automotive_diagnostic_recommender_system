import pickle
import keras
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import register_keras_serializable
from keras.layers import Layer
from data_preprocessing import process_data_for_predictions
from cvf_da_model_mod import CrossConv2D, encode_categorical_features, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

def recommend_with_cvf_da(data):
    # Load the trained model
    cvf_da_model = keras.models.load_model('./model-data/cvf_da_fully_trained.keras')

    # Prepare data that we are recommending for
    prepared_data = process_data_for_predictions(data)
    prepared_data = encode_categorical_features(prepared_data, LabelEncoder())
    prepared_data_input = [prepared_data[feature].values for feature in CATEGORICAL_FEATURES] + \
                          [prepared_data[NUMERICAL_FEATURES].values]

    # Generate recommendation
    # The output is the probabilities for each class
    cvf_da_predictions = cvf_da_model.predict(prepared_data_input)

    # To convert these probabilities into actual class predictions, we take the diagnostic action with
    # the highest probability.
    predicted_actions = np.argmax(cvf_da_predictions, axis=1)

    # Convert these encoded class labels back to the original labels with the LabelEncoder that was
    # used to fit the 'otxsequence'.
    with open('./model-data/label_encoder.pkl', 'rb') as file:
        loaded_label_encoder = pickle.load(file)
    predicted_actions = loaded_label_encoder.inverse_transform(predicted_actions)

    return predicted_actions
