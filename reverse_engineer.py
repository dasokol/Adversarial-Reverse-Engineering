"""
module for filtering, training, reverse engineering, and comparing experiment results
"""

import sparse_vector as sv
from nltk import word_tokenize
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

# K number of folds for k-fold cross validation, total number of emails from trec07p, and stopwords from nltk
K = 10
#NUM_EMAILS = 75419
#TODO figure out a way to parse rest of emails that follow different format
NUM_EMAILS = 26162
STOPWORDS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', \
	'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 
	'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 
	'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
	'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
	'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
	'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 
	'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
	'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 
	'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
#TODO remove "won" from list?

def get_relevant_content(data, num):
	"""
	relevant content is email subject and content
	return subject + content
	"""
	# split data after subject into 2 parts and grab content, assert that "Lines: " appears exactly once
	data = data.split("Lines: ")
	# hardcoded because email 26162 doesn't have a "Lines: " field
	if len(data) == 1: data = data[0].split("tests=BAYES_00,DK_POLICY_SIGNSOME,FORGED_RCVD_HELO")[-2:]
	if len(data) != 2: print(num)
	assert(len(data) == 2)
	content = " ".join(data[1].splitlines()[1:])

	# split data into 2 parts and grab subject, assert that "Subject: " appears exactly once
	data = data[0].split("Subject: ")
	if len(data) == 1:
		subject = ""
	else:
		#TODO message 4 and 117 have two subjects
		if len(data) != 2: data = data[-2:]
		assert(len(data) == 2)
		subject = data[1].splitlines()[0]
	
	return subject + " " + content

def get_emails():
	"""
	read trec email data, filtering out everything except subject and content
	return [(label, content), ...]
	"""
	print("Loading emails...")
	emails = []
	
	for email_num in range(1, NUM_EMAILS + 1):
		with open("trec07p/data/inmail." + str(email_num), "rb") as file:
			# read file as binary, then convert to string to avoid encoding errors
			data = str(file.read())[2:-1]

		content = get_relevant_content(data, email_num)
		emails.append(content)

	print("Emails loaded.")

	return emails

def get_labels():
	"""
	return a list of labels in order of email_num, True if spam (positive instance) False if ham (negative instance)
	"""
	labels = []
	
	with open("trec07p/full/index", "r") as file:
		for line in file:
			# grab and append label, assert it is either spam or ham
			label = line.split(" ")[0]
			assert(label == "spam" or label == "ham")
			label = True if label == "spam" else False
			labels.append(label)

	return labels
		
def process_emails(emails):
	"""
	modify email content to tokenize, filter stopwords, etc
	modifies emails
	"""
	print("Processing emails...")

	#TODO nltk tokenize, stopwords, stemming or lemmatizing, FreqDist?, remove most common words?, convert to lowercase or not?
	for i in range(len(emails)):
		# tokenize, remove stopwords, and convert to lowercase
		emails[i] = [word.lower() for word in word_tokenize(emails[i]) if word not in STOPWORDS]

	print("Emails processed.")

def get_word_counts(emails):
	"""
	count up occurrences of each word in each email that appears in corpus
	return [[0, 1, 0, 0, 2, ...], ...]
	"""
	print("Counting words...")

	# parse each word in each email and place into a set (set comprehension), then map each element to a unique index
	#TODO figure out a way to use tf idf or something to pick top thousand or so features
	words_to_indexes = list({word for email in emails for word in email})[:10000]
	words_to_indexes = {words_to_indexes[i]:i for i in range(len(words_to_indexes))}
	word_counts = []

	for email in emails:
		email_count = sv.SparseVector(len(words_to_indexes), 0)
		for word in email:
			# ignore word if not a feature
			try:
				email_count[words_to_indexes[word]] = email.count(word)
			except KeyError:
				continue

		word_counts.append(email_count)

	print("Words counted.")

	return word_counts

def main():
	# read in data and labels
	emails = get_emails()
	labels = get_labels()

	# process, count, and clean data
	process_emails(emails)
	word_counts = get_word_counts(emails)
	del emails

	kf = KFold(n_splits=K)
	scores = []

	for training_indexes, testing_indexes in kf.split(word_counts):
		# split training and testing data/labels
		training_data = [word_counts[i] for i in training_indexes]
		training_labels = [labels[i] for i in training_indexes]
		testing_data = [word_counts[i] for i in testing_indexes]
		testing_labels = [labels[i] for i in testing_indexes]

		mnb = MultinomialNB()
		mnb.fit(training_data, training_labels)
		scores.append(mnb.score(testing_data, testing_labels))

	accuracy = str(sum(scores) / K)
	print("Average accuracy after 10-fold cross validation: " + accuracy)
	
	#TODO only look at top 15? most influential tokens for each email

if __name__ == "__main__":
	main()
