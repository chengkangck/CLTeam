from Corpus import *

if __name__ == '__main__':
	
	#Read corpus
	#corp_path = raw_input("Please input the path of the corpus:\n")
	#train_file = raw_input("Please input the filename of the training data:\n")
	#gold_file = raw_input("Please input the filename of the gold label:\n")
	train_file = "trail.csv"
	gold_file = "trial.labels"
	corpus = Corpus(train_file)
	#corpus.readCourpus()
	
	#training part.
	'''..to be complete
	
	'''

	predict_file = "trial.predict"
	
	#Evaluation
	if (corpus.gold_file != gold_file):
		corpus.readGold(gold_file)
	if (corpus.predict_file != predict_file):
		corpus.readPrediction(predict_file)
    	corpus.evaluation()
   	corpus.print_result()