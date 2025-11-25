import pandas as pd


songs = pd.read_csv("features_30_sec.csv")
state = 1337


genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

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


