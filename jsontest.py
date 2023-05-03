import json

jsonFile = open("singlejson.jsonl", 'r')
values = json.load(jsonFile)
jsonFile.close()


print(values['clean_text'])
