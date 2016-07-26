# PFM - patient forum miner
# create_summaries_for_unseen_data_TNO.py

A conversion script. It reads the json output of the semantic search engine, applies an extractive summarization model (linear model for post selection), and prints json output.

python3 create_summary_for_unseen_data_TNO.py example_query_result_full_threads_improved.json example_query_result_full_threads.summary.json

 + 1. Read json file (query+result list), and extract threads
 + 2. For each thread in result list, extract post feats
 + 3. Standardize post feats
 + 4. Apply linear model
 + 5. Apply threshold
 + 6. Write to json file with for each thread, for each postid the value 1 or 0 for in/out summary, and the predicted value of the linear model
