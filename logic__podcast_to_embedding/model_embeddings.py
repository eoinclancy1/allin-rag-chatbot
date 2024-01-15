# For turning the csv training data into the embeddings
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
import pandas as pd

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Get the required documents
####  df = pd.read_csv("urls.csv")
#### urls_list = df.values.tolist()
#### urls = flat_list = [item for sublist in urls_list for item in sublist]
#### print(urls)

# Get the loaders
#loader = CSVLoader(file_path='clean_transcriptions/clean_phillips.csv', source_column="Speaker",csv_args={ 
#    'delimiter': ',',
#    'fieldnames': ['Topic', 'Sub-topic', 'Speaker', 'Question', 'Answer']
#})


# drop the first element in 'data' which just holds the first row of titles 
#data = loader.load()[1:]
# print(data)
documents = []

df = pd.read_csv('clean_transcriptions/clean_christie.csv')
# df = pd.read_csv('clean_transcriptions/clean_christie.csv', encoding='unicode_escape') sometimes needed

# I think I should split in some way like this where 
# we pass the topic, sub-topic, speaker and question in as the meta data, and then split up the answer up into 
# potentially multiple docs that have some overlap - https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/markdown_header_metadata 

# Start to chunk the text
# Good reference doc: https://www.pinecone.io/learn/chunking-strategies/

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 100
)

for index, row in df.iterrows():
    metadata = {
        'source': row['Speaker'] + ' - Question: ' + str(index),
        'row': index,
        'topic': row['Topic'],
        'sub-topic': row['Sub-topic'],
        'question': row['Question']
    }

# https://stackoverflow.com/questions/76603417/typescript-langchain-add-field-to-document-metadata    
    try:
        docs = text_splitter.create_documents([row['Answer']], [metadata])
        documents.extend(docs)
        print(docs)
    except:
        print("An exception occurred")


print('Length of the embeddings is: ', len(documents))

# Run the embeddings
embeddings = OpenAIEmbeddings()
vectorStore_openAI = FAISS.from_documents(documents, embeddings)

# store the embeddings - https://python.langchain.com/docs/integrations/vectorstores/faiss
vectorStore_openAI.save_local("faiss_index_christie")

# load the embeddings
loaded_vectorStore = FAISS.load_local("faiss_index_christie", embeddings)


#### Three lightweight tests of the embedding - Customize as you want ####

# Type 1: Pose question without the llm. Just referring to vector source
query = "what name does the candidate perfer to go by?"
ans = loaded_vectorStore.similarity_search(query)
print(ans[0].page_content)


# Type 2: OOTB Q&A style chain that can lookup a vector db and provide the best result (only 1) as input to the LLM to answer the question
# Prepare the Q&A bot - https://api.python.langchain.com/en/stable/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html?highlight=retrievalqawithsourceschain
llm = OpenAI(temperature=0,)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=loaded_vectorStore.as_retriever())
query = "what political party is he running on behalf of?"
result = chain({"question": query}, return_only_outputs=True)
print(result)

# Type 3: Assumption is that more sources fed would be better. Here we can just check what the other sources would have been, and if they'd be beneficial - likely yes!
# Improved Q&A bot - https://stackoverflow.com/questions/76482024/how-to-get-more-detailed-results-sources-with-langchain
#query = "what is the candidate's opinion on the US involvement in the Ukraine-Russia war?"
results = loaded_vectorStore.similarity_search_with_score(query)

