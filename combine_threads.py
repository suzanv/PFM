# coding=utf-8
# python3 create_summary_for_unseen_data_TNO.py json_example_query_results.new.json json_example_query_results.summary.json Dutch_model.json
# python3 create_summary_for_unseen_data_TNO.py json_example_query_results.new.json json_example_query_results.summary.json English_model.json

#0. Read the config file with models and thresholds (json, 3rd argument)
#1. Read json output of semantic search engine (query+result list), and extract threads
#2. For each thread in result list, extract post feats and sentence feats
#3. Standardize features
#4. Apply linear models
#5. By default, include half of the sentences (predicted value > median for sentences) and half of the posts (predicted value > median for posts)
#6. Write to json file with for each thread, for each postid and for each sentence the value 1 or 0 for in/out summary, and the predicted value of the linear model.


import sys
import re
import string
import functools
import operator
import numpy
import json
from scipy import sparse
import scipy
from scipy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
import time

outfilename = sys.argv[1]

threads = []
import glob
for file in glob.glob("*[0-9].json"):
	print(file)
	json_string = ""
	with open(file,'r') as json_file:
		for line in json_file:
			json_string += line.rstrip()
	parsed_json = json.loads(json_string)
	threads.append(parsed_json)


json_out = open(outfilename, 'w')
json.dump(threads,json_out)
json_out.close()
	
print (time.clock(), "\t", "thread printed")


