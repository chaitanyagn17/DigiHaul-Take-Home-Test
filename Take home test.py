# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 

@author: Garimella Naga Chaitanya
"""

import pandas as pd
import numpy as np
from dateutil import parser
from datetime import datetime, timedelta
import joblib

# Load data
shipments = pd.read_csv('Shipment_bookings.csv')
gps_data = pd.read_csv('GPS_data.csv')

def custom_date_parser(date_string):
    try:
        return parser.parse(date_string)
    except Exception as e:
        print(f"Error parsing date: {e}")
        return pd.NaT

shipments['LAST_DELIVERY_SCHEDULE_LATEST'] = shipments['LAST_DELIVERY_SCHEDULE_LATEST'].apply(custom_date_parser)
shipments['LAST_DELIVERY_SCHEDULE_LATEST'] = shipments['LAST_DELIVERY_SCHEDULE_LATEST'].dt.strftime('%Y-%m-%d %H:%M:%S')

gps_data['RECORD_TIMESTAMP']  = gps_data['RECORD_TIMESTAMP'].apply(custom_date_parser)
gps_data['RECORD_TIMESTAMP']= pd.to_datetime(gps_data['RECORD_TIMESTAMP'], errors='coerce',utc=True)
gps_data['RECORD_TIMESTAMP']  = gps_data['RECORD_TIMESTAMP'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Filter shipment bookings between October 1st and December 31st, 2023
start_date = '2023-10-01'
end_date = '2023-12-31'
filtered_shipments = shipments[(shipments['LAST_DELIVERY_SCHEDULE_LATEST'] >= start_date) &
                              (shipments['LAST_DELIVERY_SCHEDULE_LATEST'] <= end_date)]


# Get the last GPS record for each shipment
latest_gps = gps_data.sort_values(by='RECORD_TIMESTAMP').groupby('SHIPMENT_NUMBER').last().reset_index()

# Merge the GPS data with the shipment data
merged_data = filtered_shipments.merge(latest_gps, on='SHIPMENT_NUMBER', how='left', suffixes=('_scheduled', '_actual'))
merged_data['LAST_DELIVERY_SCHEDULE_LATEST'] = pd.to_datetime(merged_data['LAST_DELIVERY_SCHEDULE_LATEST'] , errors='coerce',utc=True)

# Calculate the on-time status
merged_data['ON_TIME_DELIVERY'] = merged_data['RECORD_TIMESTAMP'] <= (merged_data['LAST_DELIVERY_SCHEDULE_LATEST'] + timedelta(minutes=30))

'''
Operational teams rely on KPIs like on-time collection and on-time delivery to gauge carrier
performance. What percentage of shipments met the on-time delivery threshold (arriving no
later than 30 minutes past the scheduled delivery window) between October 1st and
December 31st, 2023? Please outline your assumptions.

Assumptions: 
    
   1. A shipment is considered on-time if it arrives no later than 30 minutes past the 
      latest scheduled delivery time (LAST_DELIVERY_SCHEDULE_LATEST).
   2. We will filter the shipments based on the given date range, i.e., from October 1st to December 31st, 2023.
   3.GPS data will be used to determine the actual delivery times.

'''

# Calculate the percentage of on-time deliveries
on_time_percentage = merged_data['ON_TIME_DELIVERY'].mean() * 100
print(f"Percentage of on-time deliveries: {on_time_percentage:.2f}%")


''' Predict the likelihood of delay for the list of shipments in “New_bookings.csv” dataset.'''

''' Utilise additional data sources by making API calls. - Weather data and Traffic Data '''
import requests

# Replace with your actual API keys
WEATHER_API_KEY = 'wether_key'
# TRAFFIC_API_KEY = 'traffic_key'

WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'
# TRAFFIC_API_URL = 'https://api.tomtom.com/map/1/tile/basic/main'

def fetch_weather_data(lat, lon, timestamp):
    params = {
        'lat': lat,
        'lon': lon,
        'appid': WEATHER_API_KEY,
        'units': 'metric',
        # 'dt': int(timestamp.timestamp())  # convert to UNIX timestamp if needed
    }
    response = requests.get(WEATHER_API_URL, params=params)
    if response.status_code == 200:
        data = response.json()
        return {
            'weather_main': data['weather'][0]['main'],
            'temperature': data['main']['temp'],
            'wind_speed': data['wind']['speed']
        }
    else:
        return {'weather_main': 'Unknown', 'temperature': None, 'wind_speed': None}

# def fetch_traffic_data(lat, lon, timestamp):
#     params = {
#         'lat': lat,
#         'lon': lon,
#         'apikey': TRAFFIC_API_KEY,
#         # 'timestamp': timestamp.isoformat()
#     }
#     response = requests.get(TRAFFIC_API_URL, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         return {
#             'traffic_congestion': data.get('congestion', 'Unknown'),
#             'traffic_speed': data.get('speed', None)
#         }
#     else:
#         return {'traffic_congestion': 'Unknown', 'traffic_speed': None}

def add_external_data(df):
    def fetch_data(row):
        weather_data = fetch_weather_data(row['FIRST_COLLECTION_LATITUDE'], row['FIRST_COLLECTION_LONGITUDE'], row['FIRST_COLLECTION_SCHEDULE_EARLIEST'])
        # traffic_data = fetch_traffic_data(row['FIRST_COLLECTION_LATITUDE'], row['FIRST_COLLECTION_LONGITUDE'], row['FIRST_COLLECTION_SCHEDULE_EARLIEST'])
        return {**weather_data}#,**traffic_data}
    
    external_data = df.apply(fetch_data, axis=1)
    external_df = pd.json_normalize(external_data)
    return pd.concat([df, external_df], axis=1)

merged_data = add_external_data(merged_data)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load new bookings data
new_bookings = pd.read_csv('New_bookings.csv')
new_bookings = add_external_data(new_bookings)
# Feature engineering (example features)
def create_features(df):
    df['DELIVERY_WINDOW'] = (pd.to_datetime(df['LAST_DELIVERY_SCHEDULE_LATEST']) - pd.to_datetime(df['LAST_DELIVERY_SCHEDULE_EARLIEST'])).dt.total_seconds() / 3600
    df['COLLECTION_WINDOW'] = (pd.to_datetime(df['FIRST_COLLECTION_SCHEDULE_LATEST']) - pd.to_datetime(df['FIRST_COLLECTION_SCHEDULE_EARLIEST'])).dt.total_seconds() / 3600
    df['DISTANCE'] = ((df['LAST_DELIVERY_LATITUDE'] - df['FIRST_COLLECTION_LATITUDE'])**2 + (df['LAST_DELIVERY_LONGITUDE'] - df['FIRST_COLLECTION_LONGITUDE'])**2)**0.5
    return df

# Prepare training data
train_data = create_features(merged_data)
new_bookings = create_features(new_bookings)


# Custom transformation function for multiple variables
def handle_unknown_categories(X, categories_dict):
    X_transformed = X.copy()
    for column in X.columns:
        X_transformed[column] = X[column].apply(lambda x: x if x in categories_dict[column] else 'others')
    return X_transformed


features = ['VEHICLE_SIZE', 'VEHICLE_BUILD_UP','DELIVERY_WINDOW' ,'COLLECTION_WINDOW','DISTANCE','weather_main','temperature','wind_speed']
target = ['ON_TIME_DELIVERY']


# Extract unique categories for each feature
unique_categories = {col: train_data[col].unique() for col in train_data.columns if col != 'target'}

# Create a FunctionTransformer with the custom function
transformer = FunctionTransformer(handle_unknown_categories, kw_args={'categories_dict': unique_categories})

preprocessor = ColumnTransformer(
    transformers=[
        ('VEHICLE_SIZE', Pipeline(steps=[
            ('handle_unknown', FunctionTransformer(handle_unknown_categories, kw_args={'categories_dict': unique_categories})),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), ['VEHICLE_SIZE']),
        ('VEHICLE_BUILD_UP', Pipeline(steps=[
            ('handle_unknown', FunctionTransformer(handle_unknown_categories, kw_args={'categories_dict': unique_categories})),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), ['VEHICLE_BUILD_UP']),
        ('weather_main', Pipeline(steps=[
            ('handle_unknown', FunctionTransformer(handle_unknown_categories, kw_args={'categories_dict': unique_categories})),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), ['weather_main']),
        
        ('num', FunctionTransformer(), ['DELIVERY_WINDOW', 'COLLECTION_WINDOW', 'DISTANCE','temperature','wind_speed'])
    ])


# Create a pipeline with preprocessor and classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=43))
])

# Split the data into training and test sets
X = train_data[features]

y = train_data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# To get the features which are impacting for delay
model = pipeline.named_steps['classifier']

# Get feature importances
importances = model.feature_importances_

# Get the feature names after preprocessing
# Extract the transformed feature names (including one-hot encoded features)
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out

# Create a DataFrame for the feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)



# Predict on the new bookings data
new_bookings_predictions = pipeline.predict(new_bookings[features])
new_bookings_probabilities = pipeline.predict_proba(new_bookings[features])[:,1]

new_bookings['DELAY'] = new_bookings_predictions
new_bookings['DELAY_PROBABILITY'] = new_bookings_probabilities

# Save the predictions
new_bookings[['SHIPMENT_NUMBER','DELAY' ,'DELAY_PROBABILITY']].to_csv('New_bookings_predictions.csv', index=False)

# Save the model
joblib.dump(pipeline, 'model.pkl')


