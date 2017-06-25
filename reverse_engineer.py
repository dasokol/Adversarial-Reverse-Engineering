"""
module for filtering, training, reverse engineering, and comparing experiment results
"""

NUM_EMAILS = 75419

def get_labels():
	"""
	return a list of labels in order of email_num
	"""
	labels = []
	
	with open("trec07p/full/index", "r") as file:
		for line in file:
			# grab and append label, assert it is either spam or ham
			label = line.split(" ")[0]
			assert(label == "spam" or label == "ham")
			labels.insert(label)

	return labels
		
def get_relevant_content(data):
	"""
	relevant content is email subject and content
	return subject + content
	"""
	# split data into 2 parts and grab subject, assert that "Subject: " appears exactly once
	data = data.split("Subject: ")
	assert(len(data) == 2)
	subject = data[1].splitlines()[0]
	
	# split data after subject into 2 parts and grab content, assert that "Length: " appears exactly once
	data = data[1].split("Length: ")
	assert(len(data) == 2)
	content = " ".join(data[1].splitlines()[1:])

	return subject + " " + content

def get_emails_and_labels():
	"""
	read trec email data, filtering out everything except subject and content, and pair with labels
	return [(label, content), ...]
	"""
	labels = get_labels()
	emails = []
	
	for email_num in range(1, NUM_EMAILS + 1):
		with open("trec07p/data/inmail." + str(email_num), "r") as file:
			data = file.read()

		content = get_relevant_content(data)
		emails.append((labels[email_num], content))

	return emails

def main():
	emails = get_emails_and_labels()

if __name__ == "__main__":
	main()
