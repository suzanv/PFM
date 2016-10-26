# PFM - patient forum miner
## create_summaries_for_unseen_data_TNO.py

This script summarizes all threads retrieved for a query. It reads the json output of the semantic search engine, applies two extractive summarization models (linear model for post selection and linear model for sentence selection), and prints json output.

```
python3 create_summary_for_unseen_data_TNO.py example_query_result_full_threads_improved.json example_query_result_full_threads.summary.json Dutch_model.json
```

 + 0. Read the config file with models and thresholds (json, 3rd argument)
 + 1. Read json output of semantic search engine (query+result list), and extract threads
 + 2. For each thread in result list, extract post feats and sentence feats
 + 3. Standardize features
 + 4. Apply linear models
 + 5. Apply thresholds
 + 6. Write to json file with for each thread, for each postid and for each sentence the value 1 or 0 for in/out summary, and the predicted value of the linear model.
 

The runtime of the script is linear with the number of sentences. The function that costs the most time is the calculation of the cosine similarities. On average, feature extraction and summarization together takes 1.5 second per thread.
