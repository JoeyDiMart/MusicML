import pandas as pd
from sklearn.preprocessing import LabelEncoder  # turns string labels into numeric labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # for standardization aka z-score normalization (removes the mean and deviations allowing for less bias)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import librosa  # library for video and audio analysis
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
'''
neural_sizes = [
    (100, 50)  # kept this neural network size as a list just to keep the code relatively the same and able to test later
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

model = max(results, key=lambda x: x['accuracy'])['model']  # make sure the model used is the most accurate (if testing multiple)


######## Below here is where we handle taking a .wav file and predicting the genre ########
audio_path = input("enter the audio file name (make sure it's in the same directory): ")

y, sr = librosa.load(audio_path, duration=30)  # duration 30 for a 30 second clip max

features = {}  # librosa will take the .wav file and extract all useful components of it (below)
# these features also match the "features_30_sec.csv" file

features['length'] = len(y)

# 1. Chroma STFT
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
features['chroma_stft_mean'] = np.mean(chroma_stft)
features['chroma_stft_var'] = np.var(chroma_stft)

# 2. RMS (Root Mean Square Energy)
rms = librosa.feature.rms(y=y)
features['rms_mean'] = np.mean(rms)
features['rms_var'] = np.var(rms)

# 3. Spectral Centroid
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
features['spectral_centroid_mean'] = np.mean(spectral_centroid)
features['spectral_centroid_var'] = np.var(spectral_centroid)

# 4. Spectral Bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
features['spectral_bandwidth_var'] = np.var(spectral_bandwidth)

# 5. Spectral Rolloff
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
features['rolloff_mean'] = np.mean(rolloff)
features['rolloff_var'] = np.var(rolloff)

# 6. Zero Crossing Rate
zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
features['zero_crossing_rate_var'] = np.var(zero_crossing_rate)

# 7. Harmony and Perceptr (Percussive)
harmony, perceptr = librosa.effects.hpss(y)
features['harmony_mean'] = np.mean(harmony)
features['harmony_var'] = np.var(harmony)
features['perceptr_mean'] = np.mean(perceptr)
features['perceptr_var'] = np.var(perceptr)

# 8. Tempo
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
features['tempo'] = tempo

# 9. MFCCs (20 coefficients)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
for i in range(1, 21):
    features[f'mfcc{i}_mean'] = np.mean(mfccs[i - 1])
    features[f'mfcc{i}_var'] = np.var(mfccs[i - 1])

# print(features)  # the features match the .csv file correctly



######## Below is the code to predict the genre of the uploaded .wav file ########

feature_df = pd.DataFrame([features])

# Get the column order from your training data (X before dropping filename and label)
training_columns = songs.drop(['filename', 'label'], axis=1).columns

feature_df = feature_df[training_columns]  # put into a dataframe and match the training columns

features_scaled = scaler.transform(feature_df)  # scale using same scalar as training

prediction = model.predict(features_scaled)  # predict (which is able to use the best model we train above

genre = le.inverse_transform(prediction)  # this is how we change numerical label back to the genres we know

print(f"Predicted genre: {genre[0]}")