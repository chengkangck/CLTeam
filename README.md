# NLP_teamlab_SS18

## WASSA 2018 Implicit Emotion Shared Task

Introduction
Emotion is a concept that is challenging to describe. Yet, as human beings, we understand the emotional effect situations have or could have on us and other people. How can we transfer this knowledge to machines? Is it possible to learn the link between situations and the emotions they trigger in an automatic way?

In the light of these questions, we proposed the Shared Task on Implicit Emotion Recognition, organized as part of WASSA 2018 at EMNLP 2018 aims at developing models which can classify a text into one of the following emotions: Anger, Fear, Sadness, Joy, Surprise, Disgust without having access to an explicit mention of an emotion word.

Task Description
Participants were given a tweet from which a certain emotion word is removed. That word is one of the following: "sad", "happy", "disgusted", "surprised", "angry", "afraid" or a synonym of one of them. The task was to predict the emotion the excluded word expresses: Sadness, Joy, Disgust, Surprise, Anger, or Fear.

With this formulation of the task, we provide data instances which are likely to express an emotion. However, the emotion needs to be inferred from the causal description, which is typically more implicit than an emotion word. We therefore presume that successful systems will take into account world knowledge in a structured or statistical manner.

Examples are:

"It's [#TARGETWORD#] when you feel like you are invisible to others."
"My step mom got so [#TARGETWORD#] when she came home from work and saw
that the boys didn't come to Austin with me."
"We are so #[#TARGETWORD#] that people must think we are on good drugs
or just really good actors."
The shared task consisted of the challenge to build a model which recognizes that [#TARGETWORD#] corresponds to sadness ("sad") in the first two examples and with joy ("happy") in the third.
