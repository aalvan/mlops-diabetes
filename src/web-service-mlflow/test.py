import requests

features = {
    	'Pregnancies':6, 
        'Glucose':148,
        'BloodPressure':72,
        'SkinThickness':35,
        'Insulin':0,
        'BMI':33.6,
        'DiabetesPedigreeFunction':0.627,
        'Age':50}

url = 'http://127.0.0.1:9696/predict'

response = requests.post(url, json=features)
print(response.json())