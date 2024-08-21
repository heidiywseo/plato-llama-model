import pickle
from sklearn.model_selection import train_test_split

with open('tokenized_data.pkl', 'rb') as f:
    tokenized_data, pad_id = pickle.load(f)

def split_data(tokenized_data, train_size=0.8, valid_size=0.1):
    train_data, temp_data = train_test_split(tokenized_data, train_size=train_size, random_state=42)
    valid_data, test_data = train_test_split(temp_data, train_size=valid_size/(1-train_size), random_state=42)
    return train_data, valid_data, test_data

train_data, valid_data, test_data = split_data(tokenized_data)

with open('split_data.pkl', 'wb') as f:
    pickle.dump((train_data, valid_data, test_data, pad_id), f)

