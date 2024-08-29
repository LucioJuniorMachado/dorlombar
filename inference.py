import joblib

model = joblib.load('model/spine_model.pkl')


import pandas as pd
data = pd.read_csv('data/Dataset_spine_unknown.csv')

inferences = model.predict(data)

print(inferences)

data['previsoes'] = inferences

data.to_csv( 'model/inferences.csv', index = False)


