"""
    Text Summarization using IDF

    Authors: David Yang, Kenny Gonzalez, Joram Amador
"""
#progress bar lol
import json
from tqdm import tqdm

import argparse
from collections import Counter
import math

from rouge import Rouge

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# some seaborn stuff
sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes
    

# needs _ arguments
parser = argparse.ArgumentParser()
parser.add_argument("training_data", help="data to train the model and get IDFs for words")
parser.add_argument("test_data", help="data to run the model on and output sample summaries")
parser.add_argument("numSent", help="number of sentences to be output")
parser.add_argument("rho", help="value to be added to sum if word does not have IDF")
# parses arguments into args
args = parser.parse_args()

# save args
training_data = args.training_data
test_data = args.test_data
numSent = int(args.numSent)
rho = float(args.rho)

stoplistFile = "./data/stoplist.txt"
sList = open(stoplistFile, "r", encoding="utf-8")
stoplist = set(sList.readlines())
stoplist = [word.strip("\n") for word in stoplist]

def training(trainingFile, stoplist):
    """
        Trains the model by going through the training 
        documents and calculating the IDFs for the words

        Returns:
            IDFs (dict): dictionary containing the IDFs for each word

        Parameters:
            trainingFile (str): path to file with the documents to train on
            stoplist (set): set of words to not be considered in calculations
    """
    wordCount = dict()
    
    training_docs = open(trainingFile, "r", encoding="utf-8")

    doc_count = 0

    # get df for all words
    for doc in tqdm(training_docs):
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

def avgIDF (IDFs, testBillString, stoplist, rho):
    """
        Gets the average IDF for each sentence in a test doc

        Returns:
            avg_sent_IDFs (dict): dictionary containing the average IDFs for each sentence

        Parameters:
            IDFs (dict): dictionary containing IDFs of words from training
            testBillString (str): test bill as a string
            stoplist (set): set of words to not be considered in calculations
            rho (float): value to be added to sum if word not seen in training
    """
    # test_doc = open(testFile, "r", encoding="utf-8")
    avg_sent_IDF = dict()

    sum = 0
    # will only iterate once
    # for doc in test_doc:
    for sentence in testBillString.split(". "): # split each doc into sentences
        # print(sentence)
        if len(sentence.split(' ')) > 4:
            for word in sentence.split():
                # word in IDFs and word not in stoplist
                if word.lower() in IDFs and word.lower() not in stoplist:
                    sum += IDFs[word.lower()]
                else:
                    sum += rho
            avg = sum / len(sentence)
            avg_sent_IDF[f'{" ".join(sentence.split())}.'] = avg
    # print(avg_sent_IDF) 
    # test_doc.close()
    return avg_sent_IDF

def outputSummary (avg_sentence_IDFs, numSentences=5):
    """
        outputs summary into a new file for a doc given the avg sentence IDFs of that doc

        Parameters:
            avg_sentence_IDFs (dict): dictionary containing avg IDFs of sentences from test doc
            numSentences (float): how long the summary should be
    """
    output_string = ""
    output_file = open("./data/output_summaries", "w")
    for sentence in (Counter(avg_sentence_IDFs).most_common(numSentences)):
        # make sure that the summary stays on one line (since the sentence has \n)
        output_string += sentence[0].strip() + ' '
    output_string.strip()
    output_file.write(output_string)
    output_file.close()



def rougeVSSentences(maxSentences):

    # this should pull the summary from each line in the us_test_data_final_OFFICIAL.jsonl
    jsonFile = open("./data/us_test_data_final_OFFICIAL.jsonl", 'r')

    # opening the file of 10 bills to graph
    bills = open('./data/test_data_SHORT.txt')

    # train the IDFs dict/train the model
    IDFs = training(training_data, stoplist)

    rouge2FScores = []  # r2 f1 scores
    rouge2RScores = []  # r2 recall 
    rouge2PScores = []  # r2 precision
    rougeLFScores = []  # rL f1
    rougeLRScores = []  # rL recall 
    rougeLPScores = []  # rL precision

    # for each bill, we want to call avgIDF
    # and we want to pass in the trained IDFs dict
    # that call to avgIDFs is going to return the sentences and the avgIDFs
    # using the dict of sentences, we can call outputSummaries 

    for i in range(10): # 10 bills
        jsonBill = json.loads(jsonFile.readline())
        reference = jsonBill['summary'] # this will be a human-written summary that should be matched to bills 1-10 of test_data_SHORT.txt, we'll use it as ref for ROUGE
        
        bill = bills.readline() # this will pull the cleaned bill from ./data/test_data_SHORT.txt
        sentenceIDFs = avgIDF(IDFs, bill, stoplist, rho) # populate average IDFs for each sentence in test bill

        for j in range(1, maxSentences+1): # summaries of length 1-5 sentences
        # eventually have to write a for loop to output summary for different numSentences
            outputSummary(sentenceIDFs, j) # printing to a new document ./data/output_summaries.txt

            rouge = Rouge() # creating a new Rouge object

            # opening the outputted summaries for ROUGE eval later
            h = open('./data/output_summaries')
            hypothesis = h.readline() # reading the outputted summary
            
            scores = rouge.get_scores(hypothesis, reference)


            print(scores) # extract rouge-2 and rouge-l scores and also graph 

            # creating the lists of f1, recall, and precision scores
            rouge2FScores.append(scores[0]["rouge-2"]["f"])
            rouge2RScores.append(scores[0]["rouge-2"]["r"])
            rouge2PScores.append(scores[0]["rouge-2"]["p"])

            rougeLFScores.append(scores[0]["rouge-l"]["f"])
            rougeLRScores.append(scores[0]["rouge-l"]["r"])
            rougeLPScores.append(scores[0]["rouge-l"]["p"])

            # y-axis = rouge-2 scores, x-axis - numSentences

    # [(f1, r, p) (f1, r, p), (f1, r, p)]
    rouge2 = list(zip(rouge2FScores, rouge2RScores, rouge2PScores))
    rougeL = list(zip(rougeLFScores, rougeLRScores, rougeLPScores))

    # df_rouge2 = pd.DataFrame(rouge2, index = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], columns=['F1', 'Recall', 'Precision'])
    # df_rouge2['x1'] = df_rouge2.index

    df_rougeL = pd.DataFrame(rougeL, index = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'], columns=['F1', 'Recall', 'Precision'])
    df_rougeL['x1'] = df_rougeL.index
    # df_rouge2['ColumnA'] = df_rouge2[df_rouge2.columns[1:]].apply(
    # lambda x: ','.join(x.dropna().astype(str)),
    # axis=1)
    
    # print(df_rouge2)
    
    df_rougeL = rougeL

    

    plt.figure(figsize=(10,6), tight_layout=True)
    # ax = sns.scatterplot(data=df_rouge2, x='x1', y='F1', s=45)
    # ax2 = sns.scatterplot(data=df_rouge2, x='x1', y='Recall', s=45)
    # ax3 = sns.scatterplot(data=df_rouge2, x='x1', y='Precision', s=45)

    ax = sns.scatterplot(data=df_rougeL, x='index', y='F1', s=45)
    ax2 = sns.scatterplot(data=df_rougeL, x='index', y='Recall', s=45)
    ax3 = sns.scatterplot(data=df_rougeL, x='index', y='Precision', s=45)



    ax.set(xlabel='numSentences', ylabel='rougeScores')
    
    plt.show()

    jsonFile.close()






def main():

    # # train by getting IDFs of each word
    IDFs = training(training_data, stoplist)

    # td = open('./data/test_bill1.txt')
    # test_data = td.readline()
    # get the avg IDF of each sentence based on the training

    rouge = Rouge() # creating a new Rouge object

    

    jsonFile = open("./data/us_test_data_final_OFFICIAL.jsonl", 'r')
    td = open('./data/test_data_SHORT.txt')
    # for num in range(1, numSent +1):
        # print("Number of Sentences: " + str(num))
    
    for i in range(10):

        bill = td.readline()


        # opening the outputted summaries for ROUGE eval later
        # reading the outputted summary

        jsonBill = json.loads(jsonFile.readline())
        reference = jsonBill['summary']  
        
        sentenceIDFs = avgIDF(IDFs, bill, stoplist, rho)
        outputSummary(sentenceIDFs, 6)

        h = open('./data/output_summaries')
        hypothesis = h.read() 


        scores = rouge.get_scores(hypothesis, reference)
        
        print(f'ROUGE-L SCORE for bill {i}: {scores[0]["rouge-l"]}')
        print(reference)
    # # temp for loop to print highest avg IDF sentences
    # # for sentence in (Counter(sentenceIDFs).most_common(5)):
    # #     print(sentence[0] + '\n')


    

    # # now that we have the avg IDF for each sentence, 
    # # we want to output the sentences that meet the threshold
    # outputSummary(sentenceIDFs)

    # rougeVSSentences(2)
    

if __name__ == "__main__":
    main()