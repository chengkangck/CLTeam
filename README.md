# Different Classifiers and Features for Twitter Emotion Analysis

## Abstract
In our project, we try two different types of classifiers: Multi-Class Perceptron and Na¨ıve Bayes and different features: Bag of Words and Word Embedding.  First,  we analyse our training data to get an overview of our data set, and then we in- troduce the two classifiers and features ex- traction methods separately. We found that for this data set, the Perceptron Model achieved a slightly lower (approximately 4% lower) accuracy score than the Na¨ıve Bayes Model.

## 1	Introduction
Our task is to take a twitter data set of tweets that are labelled with their corresponding emotions and make a classifier that can predict the emotion class of tweets in unseen data, similar to the task shown in (Hasan et al, 2014).

We effectively have a training corpus of raw tweets, complete with Gold Standard emotion in- formation taken as a hashtagged emotion word contained within the tweet itself. This hashtagged emotion had previously been removed from the tweet and applied as a label denoting one of 8 emo- tion categories.

The implication of this is that none of the train- ing tweets have their corresponding emotion ex- plicitly noted as an explicit within the tweets as   a hashtag as they did originally. Therefore, the voluntary and explicit labelling of, for example, a “happy tweet” containing the indicator “#happy” is underrepresented, in our training corpus. The implication of this is that our system only has ac- cess to less explicit indicators of emotion class. We decided that our task should be to compare two classifiers because we wanted to pitch the so- phisticated machine learning technique against the simple; but surprisingly effective Na¨ıve Bayes for- mula.

##  2	Data Analysis
Analysing the given data is important for our un- derstanding of the project tasks. It makes us more familiar with the distribution of the data, in order to decide which models are more suitable for these tasks intuitively, and allow us to make design de- cisions for our classification mechanisms.

We found that there were 68 different languages in the training set, indicated by a language label, “en” for British English, “de” for German etc. We chose top 10 languages to draw a histogram shown in Figure 1. Almost all sentences were in English and Spanish, and the number of tweets in the other languages was very low. Hence, we considered the English and Spanish (“en”, “es” and “us”) lan- guages for our emotion analysis. We chose En- glish and Spanish because they were most promi- nent in the corpus. It would be a bad design deci- sion to use the other, underrepresented languages, without some translation of all tweets to the same language. This is because words that have the same meaning in different languages would be counted as separate entities where they in fact de- note the same sentiment. We wanted to avoid un- necessary low frequency counts for types, which the use of underrepresented languages would have brought. In addition, there were some language specific difficulties such as Chinese characters, for example. We also analysed the distribution of the emotion classes in the training data set, as shown in Fig- ure 2. We can clearly see that there were more than 600,000 sentences with the label “happy”. The training data has a bias within it that means that  it is inclined towards the classification of “happy”. There may be lower accuracy of emotion predic- tion because of data skew.
![image](https://github.com/chengkangck/CLTeam/blob/master/images/Figure%201The%20number%20of%20tweets%20in%20each%20language.PNG)
![image](https://github.com/chengkangck/CLTeam/blob/master/images/Figure%202The%20number%20of%20tweets%20corresponding%20to.PNG)

## 3	Different Classifiers
We tested two different classifiers for this task. One was a Perceptron Model, the other was a Na¨ıve Bayes Model. The Perceptron is a very ba- sic model which is used successfully to do clas- sifying tasks, and Na¨ıve Bayes is  a  good,  ba- sic model which is effective in text classification tasks.
### 3.1	Multi-Class Perceptron
We extended the binary classification Percep-tron to Multi-Class classification (Freund and Schapire, 1999) because there were eight differ- ent classes of emotion. The pseudo-code of the Multi-Class Classification is shown in Algorithm 1.

In Algorithm 1, we first gave each class a weight vector and initialised it by ˙0 (line 1). And in each iteration (line 2), for each training example (line 3), we predicted the emotion class of each sen- tence, using the weight vectors (line 4). If the predicted label was different from the label in the Gold Standard (line 5), we modified the weight vectors by adding the weight of the correspond- ing feature vector to the weight vector of the right class (line 6) and subtracting the corresponding feature vector for the weight vector of the pre- dicted class (line 7). Finally, we can obtain the training weight vectors for each class (line 8).

### 3.2	Na¨ıve Bayes
We made a simple Na¨ıve Bayes classifier. The fea- tures used were essentially the word list of types from the training set. To be able to accommo- date out of vocabulary words in the test set, we used Laplace smoothing as it is easy to adjust large amounts of short text and is a simple smoothing model. The formula for the Na¨ıve Bayes Model for classification is shown below:

![image](https://github.com/chengkangck/CLTeam/blob/master/images/formula1.PNG)

where y means class y, y is the predicted class, and xi represents the ith feature (word). The formula for Laplace smoothing is shown here:

![image](https://github.com/chengkangck/CLTeam/blob/master/images/formula2.PNG)


where ci means the ith class, ni is the occurrences of the word xi in the class ci, n represents the num- ber of all words in the class ci and N means the number of all words in all classes.

There are Gaussian Na¨ıve Bayes, Multinomial Na¨ıve Bayes and Bernoulli Na¨ıve Bayes models because each model type is suited to different data distributions of feature values. Because our task was focused on a large number of short texts, we thought that a binary representation of word oc-currence would represent our data set with a re- spectable degree of accuracy. Hence we chose Bernoulli Na¨ıve Bayes. In addition, we tried a Multinomial Na¨ıve Bayes model because we also used TF-IDF features.


## 4	Feature Extraction
We used two different features for our Multi-Class Perceptron Model. One was the Bag of Words, the other was Word Embedding. We also tried two different features for Na¨ıve Bayes Model: Bag of Words, and Bag of Words with TF-IDF.
### 4.1	Preprocessing
Before extracting the features, we preprocessed the text. We used regular expressions to delete words which did not hold information that was valuable for use in emotion analysis, such as: HTML tags, numbers, @tagged twitter users, for- mat indicators such as “NEWLINE” and URLs. We left the sentences in English and Spanish as de- scribed in section 2, and left all emojis. We chose to leave the emojis in the training set because these - like hashtags - are volunteered by the tweeter as an explicit representation of a sentiment expressed within the tweet, and - like emotion words - many emojis express a specific emotion. We also deleted stop words based on a list of the most frequent grammatical words, as these generally appear in tweets of every emotion and are thus not indicative of an emotion. We lemmatized all words and made them all lowercase. When our data was clean, we could proceed.
### 4.2	Bag of Words
We used a basic Bag of Words Model, which is a collection of all of the different words from all of the tweets. Bag of Words is one of the simplest language models used in NLP. It makes an uni- gram model of the text by acknowledging the oc- currence of each word. This means that our clas- sifiers ignore relations between words such as bi- grams and unigrams, as each word is represented alone, and in no specific order. Our features are es- sentially a list of tokens from the training tweets. For the Multi-Class Perceptron model, we mod- ified the basic Bag of Words Model. The number of words was very large, therefore, we had a huge number of features. This made the dimensional- ity of feature vectors for each sentence very big, hence, we chose to record the indexes of values  
as binary values in the feature vectors in order to save space consumption. Instead if a given word appears in the given document, the value of cor- responding feature vector element is 1, otherwise, the value is 0.

For the Na¨ıve Bayes Model, we also calcu- lated Term Frequency - Inverse Document Fre- quency (henceforth, TF-IDF), which is is a numer- ical statistic that is intended to reflect how impor- tant a word is to a document in a collection or cor- pus. Term frequency is the number of times that term t occurs in document d. And the inverse doc- ument frequency is the logarithmically scaled in- verse fraction of the documents that contain the word, obtained by dividing the total number of documents by the number of documents contain- ing the term, and then taking the logarithm of that quotient shown as below:

![image](https://github.com/chengkangck/CLTeam/blob/master/images/formula3.PNG)

where N is the total number of documents in the corpus and the denominator represents the number of documents where the term appears. The TF- IDF is the multiplication of term frequency and the inverse document frequency.

### 4.3	Word Embedding
The two main drawbacks of Bag of Words are  the large dimensionality and difficulty in repre- senting relationships between two words. Hence we chose another feature Word Embedding which represents word similarity by placing synonyms and relevant words closer together in the vector space (Mikolov and Chen, 2013), (Mikolov and Sutskever, 2013).

Word embedding is a mathematical embedding from a space with one dimension per word to a continuous vector space with much a lower num- ber of dimensions. The most common numbers  of dimensions are 50 or 100. Almost all existing learning methods generate the language model and word embedding at the same time. These methods include neural networks (Mikolov and Sutskever, 2013), dimensionality reduction on the word co- occurrence matrix (Lebret and Collobert, 2014), probabilistic models (Globerson, 2007), and the explicit representation in terms of the context in which words appear (Levy and Goldberg, 2014).

For the Multi-Class Perceptron model, we used the library Gensim to implement word embedding. The Perceptron can only process the feature vec- tor for each individual sentence so we changed the word vectors into sentence vectors. We calculated the average value of all words in each sentence for each dimension. In using this method, we lose a large amount of useful information, which risks making the accuracy of emotion prediction lower.

## 5	Experimental Design

The Multi-Class Perceptron with Bag of Words is the baseline accuracy score in our project. We left all of languages in this model for simplicity and implemented it from scratch, without using any li- brary related to Feature Extraction or the Percep- tron model.

Later, we used an alternative feature extraction method: Word Embedding for the Perceptron by using the library Gensim to extract features.

Finally, we used the Scikit-Learn library to im- plement the TF-IDF feature in the Na¨ıve Bayes Model. In this last experiment, we just used the English and Spanish language tweets.

### 5.1	Experimental Setting

For the Multi-Class Perceptron Classifier with the Bag of Words feature set, our training data in- cluded 1,223,467 sentences. The unseen test data set - we used “dev.csv” as our test set - comprised of 411,079 tweets.  The number of iterations was 18.  The bias was 0 and the hyper-parameter was 21. For this model with word embedding feature, we set the dimension at 100 and the number of it- erations was 21.

For the Na¨ıve Bayes Model, we deleted other languages except English and Spanish. Hence there were 979,835 sentences in the training set and 332,739 sentences in the test set. The hyper-parameter of Laplace smoothing was set 1. We tried Multinomial Na¨ıve Bayes and Bernoulli Na¨ıve Bayes. There was a little more important parameter called min df or Minimum Data Fre- quency - in the TF-IDF which means that the vo- cabulary ignore terms that have a document fre- quency strictly lower than the given threshold. When we changed this threshold from 1 to 4, the accuracy increased by approximately 15%. We also tried a unigram model and a bigram model of this TF-IDF feature.

![image](https://github.com/chengkangck/CLTeam/blob/master/images/Figure%203The%20accuracy%20of%20Training%20Data%20and%20Test.PNG)


## WASSA 2018 Implicit Emotion Shared Task

Emotion is a concept that is challenging to describe. Yet, as human beings, we understand the emotional effect situations have or could have on us and other people. How can we transfer this knowledge to machines? Is it possible to learn the link between situations and the emotions they trigger in an automatic way?

In the light of these questions, we proposed the Shared Task on Implicit Emotion Recognition, organized as part of WASSA 2018 at EMNLP 2018 aims at developing models which can classify a text into one of the following emotions: Anger, Fear, Sadness, Joy, Surprise, Disgust without having access to an explicit mention of an emotion word.

Participants were given a tweet from which a certain emotion word is removed. That word is one of the following: "sad", "happy", "disgusted", "surprised", "angry", "afraid" or a synonym of one of them. The task was to predict the emotion the excluded word expresses: Sadness, Joy, Disgust, Surprise, Anger, or Fear.

With this formulation of the task, we provide data instances which are likely to express an emotion. However, the emotion needs to be inferred from the causal description, which is typically more implicit than an emotion word. We therefore presume that successful systems will take into account world knowledge in a structured or statistical manner.

Examples are:

"It's [#TARGETWORD#] when you feel like you are invisible to others."
"My step mom got so [#TARGETWORD#] when she came home from work and saw
that the boys didn't come to Austin with me."
"We are so #[#TARGETWORD#] that people must think we are on good drugs
or just really good actors."
The shared task consisted of the challenge to build a model which recognizes that [#TARGETWORD#] corresponds to sadness ("sad") in the first two examples and with joy ("happy") in the third.
