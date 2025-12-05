import pandas as pd
from sklearn.preprocessing import LabelEncoder  # turns string labels into numeric labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # for standardization aka z-score normalization (removes the mean and deviations allowing for less bias)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import numpy as np

from sklearn.preprocessing import MinMaxScaler, RobustScaler # for testing

songs = pd.read_csv("features_30_sec.csv")
state = 1337
genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
results = []

'''
# here's a list for different neural network sizes
# 2/3 of input + output is a good starting point (80 * 2/3) + 10 = 63 (i'll use 60)
neural_sizes = [
    (60,),
    (100,),
    (100, 50),
    (100, 100),
    (150, 75, 25),
    (100, 50, 25),
    (120, 60, 30),
]
'''

# the best accuracy was 80% with (100, 50) so looking to fine tune this
neural_sizes = [
    (100, 50),
    (100, 45),
    (100, 40),
    (100, 35),
    (100, 55),
    (100, 60),
    (100, 63),
    (100, 70),
    (90, 50),
    (110, 50),
    (150, 75),
    (200, 100),
    (200, 100, 50),
    (128, 64),
    (100, 53, 50, 20, 10, 50, 100, 50)
]

### just for testing ###
# print(songs.shape)  # printed (1000, 60)
# print(songs.columns)  #included 60 columns with filenames and other song data
#print(songs['label'].value_counts)  # the label is the column that has the actual genre (they're in order from blues to rock

X = songs.drop(['filename', 'label'], axis=1)
#print(X.columns)  # verifying the non essential music feature stuff is gone
y = songs['label']  # this is 1 column of 1000 songs (100 of each genre)

le = LabelEncoder()
y_encoded = le.fit_transform(y)  # turns the 10 genre's in the genre list/labels column to values 0-9


# 80% for training 20% for testing using the X values and the y_encoded values
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=state, stratify=y_encoded
)


#scaler = StandardScaler()  # this is the scalar object where it uses the formula z = (x - u) / s
# z is the new result of the training/test data, x is the original copy of the data, u is the mean of the training data
# s is the standard deviation
#scaler = MinMaxScaler()
scaler = RobustScaler()



X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# here's a loop to test different neural network sizes
for size in neural_sizes:
    print(f"\n\nNeural Network size is: {size}")

    mlp = MLPClassifier(
        hidden_layer_sizes=size,
        max_iter=1000,
        random_state=state,
        verbose=False,  # printing messages, set to false after training ************************************
        early_stopping=True,
        alpha=0.0001,
        learning_rate_init=0.002,  # .002 had an 81 with (90, 50)
    )

    mlp.fit(X_train_scaled, y_train)
    y_prediction = mlp.predict(X_test_scaled)

    acc_score = accuracy_score(y_test, y_prediction)
    print(f"Accuracy Score: {acc_score}")

    results.append({
        'size': size,
        'accuracy': acc_score,
        'model': mlp
    })




