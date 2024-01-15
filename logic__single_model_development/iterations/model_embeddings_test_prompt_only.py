
# I believed this preceeded the majority of the single model scripts and served just as a test bed for 
# a lot of the documenation I had been reading up on.
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain import hub
from langchain.prompts import PromptTemplate


import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
import pandas as pd
from operator import itemgetter
from langchain_core.runnables import RunnableParallel

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# Run the embeddings
embeddings = OpenAIEmbeddings()
# vectorStore_openAI = FAISS.from_documents(documents, embeddings)

# store the embeddings - https://python.langchain.com/docs/integrations/vectorstores/faiss
# vectorStore_openAI.save_local("faiss_index")

# load the embeddings
loaded_vectorStore = FAISS.load_local("faiss_index", embeddings)

# test a query without the llm 
# query = "what name does the candidate perfer to go by?"
# ans = loaded_vectorStore.similarity_search(query)
# print(ans[0].page_content)


# Prepare the Q&A bot - https://api.python.langchain.com/en/stable/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html?highlight=retrievalqawithsourceschain
# Provides only 1 input to the bot
llm = OpenAI(temperature=0,)
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=loaded_vectorStore.as_retriever())
query = "why is he running for president?"
# result = chain({"question": query}, return_only_outputs=True)
# print(result)

# Improved Q&A bot
# https://stackoverflow.com/questions/76482024/how-to-get-more-detailed-results-sources-with-langchain
#query = "what is the candidate's opinion on the US involvement in the Ukraine-Russia war?"

#### Testing getting multiple source responses
# results = loaded_vectorStore.similarity_search_with_score(query) # returns 4 results by default
#print(results)
#### Return up to k results from the vector source - https://python.langchain.com/docs/use_cases/question_answering/
# retriever2 = loaded_vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# results2 = retriever2.get_relevant_documents(query)
# print(results2[0].page_content)



# Trying Method 2 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5
prompt = hub.pull("rlm/rag-prompt")
# Show the prompt if you want
# print(
#    prompt.invoke(
#        {"context": "filler context", "question": "filler question"}
#    ).to_string()
#)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

retriever = loaded_vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)


# Trying Method 3 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5 - custom prompt
template = """You are answering questions based on the transcript of a presidential candidate interview. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Don't use knowledge outside of the provided context to try and enhance the answer.
Use five sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)

rag_chain_from_docs = (
    {
        "context": lambda input: format_docs(input["documents"]),
        "question": itemgetter("question"),
    }
    | rag_prompt_custom
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda input: [doc.metadata for doc in input["documents"]],
    "answer": rag_chain_from_docs,
}

test = rag_chain_with_source.invoke(query)

print(test['answer'])






# Source for improvements - https://python.langchain.com/docs/use_cases/question_answering/
# Right now #2 has a superior answer to #3 

# Remaining steps
# 1. Try to improve the third prompt to get the best answer there, then have 3 iteratively more complex single q&a prompts
# 2. Set up the testing ground where it's easy to run multiple models together with the same query to see which is best
# 3. Create the simple UI for asking questions
# 4. Will need to be able to ask multiple questions
# 5. Then update the embeddings on another candidate - will need to add some parsing on the name, or some means of maybe manually selecting (via) UI what candidate you're interested in
# 6. Anything else then on the readme sheet.