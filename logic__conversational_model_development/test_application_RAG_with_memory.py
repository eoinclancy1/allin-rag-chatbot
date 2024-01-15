# This compares model_embeddings_prod_with_memory_chained_prompt.py to model_embeddings_prod_with_memory.py 
# to see which is best for use as the model in the actual application

from model_embeddings_prod_with_memory import method_3
from model_embeddings_prod_with_memory_chained_prompt import chained_prompt
from langchain.chains.conversation.memory import ConversationBufferWindowMemory



# Testing Queries
## Note using terms like 'he' instead of 'candidate' can lead to some odd results
queries = [
    "What is the candidates name",
    "What party is he affiliated with?",
    "What state is he from?",
    "What sort of car does he drive?",
    "Is the candidate verifiably independently wealthy?",
    "But did he generate wealth from his business ventures?",
    "What previous roles have they served in government?",
    "why is he running for president?",
    "what is the candidate's opinion on the US involvement in the Ukraine-Russia war?",
    "Tell me something interesting about the candidate",
    "What would he do about the immigration problem in the US?",
    "Does he have views on a merit-based system for it?",
    "Would the candidate reduce fiscal spending? If so, what are the top three things they'd do?"
]

print_sources_too = False
memory_single_chain = ConversationBufferWindowMemory(k=3, return_messages=True)
memory_double_chain = ConversationBufferWindowMemory(k=3, return_messages=True)

for count, query in enumerate(queries):
    print("\nQuery #" + str(count) + ": " + query)
    
    res_single_chain = method_3(query, memory_single_chain, 'dean_phillips', print_sources_too)
    memory_single_chain = res_single_chain['memory']
    print("Single Response:\n" + res_single_chain['answer'])

    
    res_double_chain = chained_prompt(query, memory_double_chain, 'dean_phillips', print_sources_too)
    memory_double_chain = res_double_chain['memory']
    print("\nChained Response:\n" + res_double_chain['answer'])
    #print(memory.load_memory_variables({}))

