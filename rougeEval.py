import json
import cleanText
from rouge import Rouge


h = open('./data/output_summaries')
hypothesis = h.read()
reference = "National Science Education Tax Incentive for Businesses Act of 2007 - Amends the Internal Revenue Code to allow a general business tax credit for contributions of property or services to elementary and secondary schools and for teacher training to promote instruction in science, technology, engineering, or mathematics ."

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference)

print(f'Rouge-2 Scores:\t {scores[0]["rouge-2"]["r"]}')

print(f'Rouge-L Scores:\t {scores[0]["rouge-l"]}')


# def pullBillsSummaries():

#     # this should pull the summary from each line in the us_test_data_final_OFFICIAL.jsonl
#     jsonFile = open("./data/us_test_data_final_OFFICIAL.jsonl", 'r')

#     # train our textSum model
#     ts = textSum()
#     ts.training()

#     for i in range(10):
#         jsonBill = json.load(jsonFile.readline())
#         summary = jsonBill['summary'] # this will be a human-written summary that should be matched to bills 1-10 of test_data_SHORT.txt







    # jsonFile.close()





