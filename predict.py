from keras.models import load_model
import pandas as pd

dc = {0:"Normal", 1:"S", 2:"V", 3:"F", 4:"Q"}
model = "/Users/h.r.rao/Downloads/tiace_model.h5"
model = load_model(model)

dataset = "data/mitbih_test.csv"
df = pd.read_csv(dataset, header=None)
feature = df.iloc[:10,:186].values
label = df[187]

X_test = feature.reshape(len(feature), feature.shape[1],1)

print(X_test.shape)

pred = model.predict(X_test)
pred = pred.argmax(axis = -1)
labels = [dc[x] for x in pred]
print(labels)