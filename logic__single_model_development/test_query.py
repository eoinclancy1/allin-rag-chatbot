from model_embeddings_prompt_testing_playground import *


# Testing Queries
## Note using terms like 'he' instead of 'candidate' can lead to some odd results
queries = [
    "What is the candidates name",
    "What party is he affiliated with?",
    "What state is he from?",
    "What sort of car does he drive?",
    "Is the candidate verifiably independently wealthy?",
    "What previous roles have they served in government?"
    "why is he running for president?",
    "what is the candidate's opinion on the US involvement in the Ukraine-Russia war?",
    "Tell me something interesting about the candidate",
    "What would he do about the immigration problem in the US?",
    "Would the candidate reduce fiscal spending? If so, what are the top three things they'd do?"
]

print_sources_too = False

for count, query in enumerate(queries):
    print("Query #" + str(count) + ": " + query + "\n")
    
    method_1(query, print_sources_too)
    method_2(query, print_sources_too)
    method_3(query, print_sources_too)
