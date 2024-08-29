api_key = "AIzaSyDpxnACoyGhPlb5G0vxhIorvl7ctf5P3Tg"
import google.generativeai as genai
import json
import time
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
completed = list()

def main(prompt):
	Source = "gemini"
	start = time.time()
	response = model.generate_content(prompt).text
	end = time.time()
	global completed
	k = { "Prompt": prompt, "Message": response, "TimeSent": start, "TimeRecvd": end, "Source": Source}
	completed.append(k)

with open ("input.txt") as f:
    for line in f:
        if not line:
            continue
        main(line.strip())

json.dump(completed, open("output.json", "w"), indent="\t")
