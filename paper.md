1. What topic did you choose? That is, what was the problem that you were working on?
- The final project is a music classifier using the GTZAN dataset. 
- This will solve the problem of not knowing what playlist you can add a song to by letting a neural network classify
them by genre


2. What data did you use? Did you collect the data yourself (if so how), or if not where did
you acquire the data?
- We're using the GTZAN dataset, which is a collection of 1000 songs with 100 songs of each genre
- there's many places to get this dataset, I got it from https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download
- in addition to this dataset, various website were used to get free .wav files of different genres


3. What model did you use, and why did you use this model? Please provide an overview
of how this model works? (That is, what is this model doing? What are the parameters
that define the model? How does it get an answer?) Feel free to include pictures if it
helps you to explain the model.
- For this project, we trained a neural network, specifically sklearn's MLPClassifier (Multi-layer Perceptron)
- MLPClassifiers work by iteratively training the model and performing partial derivatives during each step in order to 
categorize tasks. Each input (X value) is a layer (in my case there are 58), then hidden layers are defined to be the "middle"
neurons. Finally, the last hidden layer feeds into an output layer and calculates the probability for each option, in our
case there are 10 options for 10 genres. 


4. What type of preprocessing (or cleaning) did you do on the data before training your
model?
- before officially training the model I first had to drop any columns not associated with the music itself. There were 
two columns dropped, the 'labels' which had the actual genre and 'file_name' which was just "genre_00001" etc. I then
set the  Y value to be the 10 different genre options, but I used sklearn's preprocessing library to transform the string
values of the genres into numeric values 0-9. I used different scalars during training but finally decided to use sklearn's
robust scalar which removes the median values and divides the data by the IQR (difference between 75th percentile and 25th
percentile). This scalar allows for the median value to nto be affected by outliers and allow for more stable results.
- The dataset being used took 1000 .wav files and extracted useful data here: filename,length,chroma_stft_mean,
chroma_stft_var,rms_mean,rms_var,spectral_centroid_mean,spectral_centroid_var,spectral_bandwidth_mean,
spectral_bandwidth_var,rolloff_mean,rolloff_var,zero_crossing_rate_mean,zero_crossing_rate_var,harmony_mean,harmony_var,
perceptr_mean,perceptr_var,tempo,mfcc1_mean,mfcc1_var,mfcc2_mean,mfcc2_var,mfcc3_mean,mfcc3_var,mfcc4_mean,mfcc4_var,
mfcc5_mean,mfcc5_var,mfcc6_mean,mfcc6_var,mfcc7_mean,mfcc7_var,mfcc8_mean,mfcc8_var,mfcc9_mean,mfcc9_var,mfcc10_mean,
mfcc10_var,mfcc11_mean,mfcc11_var,mfcc12_mean,mfcc12_var,mfcc13_mean,mfcc13_var,mfcc14_mean,mfcc14_var,mfcc15_mean,
mfcc15_var,mfcc16_mean,mfcc16_var,mfcc17_mean,mfcc17_var,mfcc18_mean,mfcc18_var,mfcc19_mean,mfcc19_var,mfcc20_mean,
mfcc20_var,label
- when uploading a .wav file, it gets processed to fill this data 

5. Provide an overview (either in English or pseudocode) of the algorithm that is used to
train your model. (That is, to fit the model to your data). I expect that you are using a
library instead of implementing the algorithm on your own. (You are welcome to use any
library, but indicate in the paper what library you used.) You should be able to research
the algorithm used by the library for training/fitting the model. If you need help finding
information, please reach out to the professor.


6. What challenges did you encounter in doing this project? Was there trial and error
required? What did you try that didn’t work very well? What did you need to
tweak/adjust to improve your project’s performance?
- the largest challenge was all the minor tweaking that had to be done to raise the accuracy score. Initially a neural 
network of (100, 50) was the best possible with an accuracy score of 80%, which is nowhere near the target accuracy of 100.
- Tweaking included trying different scalars (standard, minmax, and robust) as well as looping through and testing different
neural network sizes, initially trying different numbers of layers of neurons, then just sticking to 2 hidden layers 
with different sizes. 
- Changing test/training size, alpha value in MLPClassifier, and number of layers did not prove to help performance at all
- helpful changes included minor tweaking of how large the two hidden layers are, changing scalars, iterations, and
learning rate.


7. How successful was your proof of concept? (That is, evaluate the final system that you
created.) What challenges/limitations (if any) prevented your proof of concept from
being more successful. Do you have any advice to a future researcher – building on your
proof of concept – on things they should do or try to improve the system or get better
results. (E.g., different data, more data, more/different computing hardware)


8. What is the potential impact of machine learning in the area that you explored? What
potential benefits could be created by machine learning systems in the area you
explored? Who would reap these benefits? (E.g., which users or types of users would
benefit?) Are there any potential risks (e.g., risks to user safety or user privacy or to
society more broadly) or concerns related to the use of machine learning in this area?
What could future researchers – or society more broadly – do to mitigate these
risks/concerns?

