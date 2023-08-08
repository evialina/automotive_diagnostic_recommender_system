import schedule
import time

db_params = {
    'dbname': 'app_db',
    'user': 'app_user',
    'password': 'password',
    'host': 'db',
    'port': '5432'
}

def fetch_diagnostic_data():
    # Establish a connection to the database
    connection = psycopg2.connect(**db_params)
    cursor = connection.cursor()

    # Build the dynamic SQL query
    select_query = sql.SQL("SELECT * FROM api.vehicles")

    cursor.execute(select_query)

    rows = cursor.fetchall()

    # Get the column names from the cursor description
    columns = [desc[0] for desc in cursor.description]

    # Create a list of dictionaries where each dictionary represents a row
    data = []
    for row in rows:
        data.append(dict(zip(columns, row)))

    # Convert the data to a JSON string
    json_string = json.dumps(data)

    cursor.close()
    connection.close()

    return json_string


def train_model(diagnostic_data):
    url = "training-service:5000/train"
    headers = {
        'Content-Type': 'application/json',
    }

    response = requests.request("POST", url, headers=headers, data=diagnostic_data)
    print(response.text)

def job():
    print("Cron job running...")

    diagnostic_data = fetch_diagnostic_data()
    
    train_model()

    print('New model trained')

# Schedule the job to run every day at 3 AM
schedule.every().day.at("03:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
