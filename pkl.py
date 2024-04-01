import pickle
from keras.models import load_model

tiace_model = "/Users/h.r.rao/Downloads/tiace_model.h5"

model = load_model(tiace_model)

pickle_file = 'tiace_pickle.pkl'

with open(pickle_file, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved to '{pickle_file}' in pickle format.")