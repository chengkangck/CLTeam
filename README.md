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
## Abstract
## Abstract
## Abstract


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
