from allennlp.predictors.predictor import Predictor
import pandas as pd

model_url = "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
predictor = Predictor.from_path(model_url)

text = "Joseph Robinette Biden Jr. is an American politician who is the 46th and\
current president of the United States. A member of the Democratic Party, \
he served as the 47th vice president from 2009 to 2017 under Barack Obama and\
represented Delaware in the United States Senate from 1973 to 2009."


def coreference(text):
    prediction = predictor.predict(document=text)
    return predictor.coref_resolved(text)

df = pd.read_csv("./data/grimms_fairytales.csv")
df = df.iloc[:, 1:]
df = df.replace(r'[\n]+', ' ', regex=True)
df = df.replace(r'[\r]+', ' ', regex=True)

df['coreference'] = df.apply(lambda row: coreference(row['Text']), axis=1)
df.to_csv('./edited.csv')



prediction = predictor.predict(document=text)  # get prediction
print("Clsuters:-")
for cluster in prediction['clusters']:
    print(cluster)  # list of clusters (the indices of spaCy tokens)
# Result: [[[0, 3], [26, 26]], [[34, 34], [50, 50]]]
print('\n\n') #Newline

print('Coref resolved: ',predictor.coref_resolved(text))  # resolved text
# Resu