import json
jsonFileName="typeInfo_finalH.json"
with open(jsonFileName,'r') as file:
    data=json.load(file)

keyPhrase = ["testPyAnnotate.py", "cal_steps.py", "ljhutil.py"]

filtered_data = [annotation for annotation in data if any(phrase in annotation["path"] for phrase in keyPhrase)]

with open("typeInfo_filtered.json","w") as file:
    json.dump(filtered_data, file, indent=4)