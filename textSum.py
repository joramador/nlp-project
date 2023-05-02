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
parser.add_argument("threshold", help="threshold of avg IDF when outputting")
# parses arguments into args
args = parser.parse_args()

# save args
training_data = args.training_data
test_data = args.test_data
start_sentences = int(args.start_sentences)
threshold = float(args.threshold)
stoplist = "data/stoplist.txt"

def training(trainingFile, stoplist):
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

    doc_count = 0
    # get df for all words
    for doc in training_docs:
        doc_count += 1
        for sentence in doc.split("."):
            # set of words seen in the current doc
            inCurrDoc = set([])
            for word in sentence.split():
                if word.lower() not in stoplist:
                    if word.lower() not in inCurrDoc:
                        inCurrDoc.add(word.lower())
                        if word.lower() not in wordCount:
                            wordCount[word.lower()] = 1
                        else:
                            wordCount[word.lower()] += 1
    
    # calculate IDF for all words
    IDFs = dict()
    for word in wordCount:
        if (wordCount[word] > 1):
            IDFs[word.lower()] = math.log(doc_count / wordCount[word.lower()])
    training_docs.close()
    return IDFs

def avgIDF (IDFs, testFile, stoplist):
    test_docs = open(testFile, "r", encoding="utf-8")
    avg_sent_IDF = dict()

    for doc in test_docs: # each line is a doc
        sum = 0
        for sentence in doc.split(". "): # split each doc into sentences
            #print(sentence)
            for word in sentence.split():
                # word in IDFs and word not in stoplist
                if word.lower() in IDFs and word.lower() not in stoplist:
                    sum += IDFs[word.lower()]
                    print(word.lower() + str(IDFs[word.lower()]))
            avg = sum / len(sentence)
            avg_sent_IDF[f'{sentence}.'] = avg
    print(avg_sent_IDF)  
    test_docs.close()
    return avg_sent_IDF

def outputSummary (avg_sentence_IDFs, threshold):
    output_string = ""
    output_file = open("output_summaries", "w")
    for sentence in avg_sentence_IDFs:
        if abs(avg_sentence_IDFs[sentence]) <= threshold:
            # make sure that the summary stays on one line (since the sentence has \n)
            output_string += sentence.strip() 
            output_string += " "
    output_string.strip()
    output_file.write(output_string)
    output_file.close()

def main():
    # train by getting IDFs of each word
    IDFs = training(training_data, stoplist)
    #print(IDFs)
    # get the avg IDF of each sentence based on the training
    sentenceIDFs=avgIDF(IDFs, test_data, stoplist)

    # now that we have the avg IDF for each sentence, 
    # we want to output the sentences that meet the threshold
    outputSummary(sentenceIDFs, threshold)

    


if __name__ == "__main__":
    main()