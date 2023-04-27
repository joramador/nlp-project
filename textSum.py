"""
    Text Summarization using IDF

    Authors: David Yang, Kenny Gonzalez, Joram Amador
"""

import argparse
import math

# needs _ arguments
parser = argparse.ArgumentParser()
parser.add_argument("training_data", help="data to train the model and get IDFs for words")
parser.add_argument("test_data", help="data to run the model on and output sample summaries")
parser.add_argument("start_sentences", help="number of sentences at start of paragraph to be weighted more")

# parses arguments into args
args = parser.parse_args()

# save args
training_data = args.training_data
test_data = args.test_data
start_sentences = int(args.start_sentences)

def training(trainingFile):
    """
        Trains the model by going through the training 
        documents and calculating the IDFs for the words

        Returns:
            IDFs (dict): dictionary containing the IDFs for each word

        Parameters:
            trainingFile (str): path to file with the documents to train on
    """
    wordCount = dict()
    
    training_docs = open(trainingFile, "r", encoding="utf-8")

    # might have to do some preprocessing here to know what counts as docs?
        # might have to lowercase all the words, maybe have a stoplist and remove the words?
        
    doc_count = 0
    # get df for all words
    for doc in training_docs:
        doc_count += 1
        inCurrDoc = set([])
        doc_words = doc.split()
        for word in doc_words:
            if word not in inCurrDoc:
                inCurrDoc.add(word)
                if word not in wordCount:
                    wordCount[word] = 1
                else:
                    wordCount[word] += 1
    
    # calculate IDF for all words
    IDFs = dict()
    for word in wordCount:
        IDFs[word] = math.log(doc_count / wordCount[word])
    return IDFs

def avgIDF (IDFs, testFile):
    test_docs = open(testFile, "r", encoding="utf-8")
    avg_sent_IDF = dict()

    for line in test_docs:
        sum = 0
        print(line)
        for word in line.split():
            if word in IDFs:
                sum += IDFs[word]
        avg = sum / len(line)
        avg_sent_IDF[line] = avg
    print(avg_sent_IDF)    

def main():
    IDFs = training(training_data)
    #print(IDFs)
    avgIDF(IDFs, test_data)

    


if __name__ == "__main__":
    main()