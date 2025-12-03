import pandas as pd
from sklearn.preprocessing import LabelEncoder  # turns string labels into numeric labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # for standardization aka z-score normalization (removes the mean and deviations allowing for less bias)


songs = pd.read_csv("features_30_sec.csv")
state = 1337
genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
results = []

# here's a list for different neural network sizes
# 2/3 of input + output is a good starting point (80 * 2/3) + 10 = 63 (i'll use 60)
neural_sizes = [
    (60,),
    (100,),
    (100, 50),
    (100, 100),
    (150, 75, 25),
    (100, 50, 25),
    (120, 60, 30)
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


scaler = StandardScaler()  # this is the scalar object where it uses the formula z = (x - u) / s
# z is the new result of the training/test data, x is the original copy of the data, u is the mean of the training data
# s is the standard deviation


X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# here's a loop to test different neural network sizes
for size in neural_sizes:


