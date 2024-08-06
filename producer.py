from confluent_kafka import Producer
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime

#function simulating sleeping between producing events
def random_sleep():
    short_sleep_probability = 0.6
    max_short_sleep = 1.0
    max_long_sleep = 5.0
    
    if random.random() < short_sleep_probability:
        sleep_time = random.uniform(0, max_short_sleep)
    else:
        sleep_time = random.uniform(max_short_sleep, max_long_sleep)
    
    time.sleep(sleep_time)


producer_conf = {
    'bootstrap.servers':'localhost:9092,localhost:9093'
}

producer = Producer(producer_conf)

streaming_data = pd.read_csv('streaming_data.csv')

streaming_data['timestamp'] = pd.NaT

for idx, row in streaming_data.iterrows():
    streaming_data.at[idx, 'timestamp'] = datetime.now()

    message = row.to_dict()
    message['timestamp'] = streaming_data.at[idx, 'timestamp'].isoformat()  

    producer.produce(topic='transactions', value=str(message))
    random_sleep()

producer.flush()
