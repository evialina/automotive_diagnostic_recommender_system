import logging
import pandas as pd
from flask import Flask, request, jsonify
from data_preprocessing import process_data_for_training
import psycopg2
from psycopg2 import sql

# Create a Flask app
app = Flask(__name__)

app.logger.setLevel(logging.DEBUG)
app.logger.addHandler(logging.StreamHandler())

db_params = {
    'dbname': 'app_db',
    'user': 'app_user',
    'password': 'password',
    'host': 'db',
    'port': '5432'
}

def fetch_warranty_data():
    # Establish a connection to the database
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    # Build the dynamic SQL query
    select_query = sql.SQL("SELECT * FROM api.claims")

    cursor.execute(select_query)

    rows = cursor.fetchall()

    columns = [desc[0] for desc in cursor.description]
    warranty_df = pd.DataFrame(rows, columns=columns)

    cursor.close()
    connection.close()

    return warranty_df

# Define the API endpoint for data preparation
@app.route("/train", methods=["POST"])
def train():
    data = request.data.decode('utf-8')

    warranty_data = fetch_warranty_data()

    train(process_data_for_training(data, warranty_data))

    return 'New model generated'

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)