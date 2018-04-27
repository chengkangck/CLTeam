class Corpus():

	def __init__(self, train_file):
		self.train_file = train_file
		self.gold_file = ""
		self.predict_file = ""
	
		self.train_set = []
		self.gold_labels = []
		self.predict_labels = []
		self.emotion = ['sad', 'joy', 'disgust', 'surprise', 'anger', 'fear']
	
		self.total_tp = self.total_tn = self.total_fp = self.total_fn = 0
		self.total_precision = self.total_recall = 0
		self.micro_f1 = self.macro_f1 = 0
		self.performance = {'sad':[0,0,0], 'joy':[0,0,0], 'disgust':[0,0,0], 'surprise':[0,0,0], 'anger':[0,0,0], 'fear':[0,0,0]}	

	
	def readCorpus(self):
		#read training data
		file = open(self.train_file)
		line = file.readline()
       		while (line != ''):
           		self.train_set.append(line.strip('\n'))
			line = file.readline()
		file.close()
		print("Successfully read corpus!")
	
	def readGold(self, gold_file):
		self.gold_file = gold_file
		
		#read gold labels
        	file = open(gold_file)
		line = file.readline()
		while (line != ''):
            		self.gold_labels.append(line.strip('\n'))
			line = file.readline()
		file.close()
		print("Successfully read gold labels!")

	
	def readPrediction(self, predict_file):
		self.predict_file = predict_file
		
		#read predicted labels
        	file = open(predict_file)
		line = file.readline()
		while (line != ''):
            		self.predict_labels.append(line.strip('\n'))
			line = file.readline()
		file.close()
		print("Successfully read predicted labels!")


	
	def evaluation(self):
		#for each emotion, count tp, fp, tn, fn and calculate its precision and recall seperatly
		for e in self.emotion: 
			tp = tn = fp = fn = 0
			accuracy = precision = recall = f1 = 0
			for i in range(len(self.gold_labels)):
				#print e, self.gold_labels[i], self.predict_labels[i]
				if self.gold_labels[i] == self.predict_labels[i] and self.gold_labels[i] == e:
					tp += 1
				elif self.predict_labels[i] == e and self.gold_labels[i] != e:
					fp += 1
				elif self.gold_labels[i] == e and self.predict_labels[i] != e:
					fn += 1
				else:
					tn += 1
				#print tp, fp, tn, fn
				#x = raw_input()
				
			if (tp + fp) != 0:
				precision = 1.0 * tp / (tp + fp) * 100
			if (tp + fn) != 0:
				recall = 1.0 * tp / (tp + fn) * 100
			if (precision + recall) != 0:
				f1 = 2.0 * precision * recall / (precision + recall)
			#saving performance for each emotion
			self.performance[e] = [precision, recall, f1]	
			
			#adding up together for further calculation	
			self.total_tp += tp
			self.total_tn += tn
			self.total_fp += fp
			self.total_fn += fn
			self.total_precision += precision
			self.total_recall += recall
			#print			
		
		#micro_f1 calculation
 		if (self.total_tp + self.total_fp) != 0:
			micro_precison = self.total_tp / (self.total_tp + self.total_fp) * 100
		if (self.total_tp + self.total_fn)!= 0:
			micro_recall = self.total_tp / (self.total_tp + self.total_fn) * 100
		if (micro_precison + micro_recall != 0):
			self.micro_f1 = 2.0  * micro_precison * micro_recall / (micro_precison + micro_recall)
			
		#macro_f1 calculation
		macro_precison = self.total_precision / len(self.emotion)
		macro_recall = self.total_recall / len(self.emotion)
		if (macro_precison + macro_recall) != 0 :
			self.macro_f1 = 2.0 * macro_precison * macro_recall / (macro_precison + macro_recall)		


    	def print_result(self):
		for e in self.emotion :
			print e, '	Precision:',self.performance[e][0], '	Recall:',self.performance[e][1],'	F1:',self.performance[e][2]
       		print "Micro F1 Score:",self.micro_f1
		print "Macro F1 Score:",self.macro_f1