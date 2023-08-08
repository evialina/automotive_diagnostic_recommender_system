import logging
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
from recommend_with_cvf_da import recommend_with_cvf_da
import psycopg2
from psycopg2 import sql
import json

# Load the saved model
# loaded_model = keras.models.load_model("./model-data/model.h5")

# Create a Flask app
app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)
app.logger.addHandler(logging.StreamHandler())

db_params = {
    'dbname': 'app_db',
    'user': 'app_user',
    'password': 'password',
    'host': 'continuous_db',
    'port': '5432'
}

def store_data(data):
    # Establish a connection to the database
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    # Get column names from the first object's keys
    json_object = json.loads(data)
    first_object = json_object[0]

    columns = list(first_object.keys())

    # Build the dynamic SQL query
    insert_query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
        sql.Identifier('continuous'),
        sql.SQL(', ').join(map(sql.Identifier, columns)),
        sql.SQL(', ').join(sql.Placeholder() * len(columns))
    )

    print(columns)
    # Loop through the array of objects and insert each row
    for obj in json_object:
        values = [obj[column] for column in columns]
        cursor.execute(insert_query, values)

    # Commit the transaction and close the connection
    connection.commit()
    cursor.close()
    connection.close()


# Define the API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    data = request.data.decode('utf-8')

    store_data(data)

    predictions = recommend_with_cvf_da(data)

    response = jsonify({"predictions": predictions.tolist()})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)