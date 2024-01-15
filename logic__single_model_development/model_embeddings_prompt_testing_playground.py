# Playground used by the unit tests, but allows you to easily edit the prompts and have the unit tests kept constant
import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain import hub, OpenAI
from langchain_core.runnables import RunnableParallel
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Just loading local embeddings
embeddings = OpenAIEmbeddings()
loaded_vectorStore = FAISS.load_local("embeddings/faiss_index_dean_phillips", embeddings)

# Method 2 & 3 Source Printing
# Finding raw sources from the Vector database
#### Return up to k results from the vector source - https://python.langchain.com/docs/use_cases/question_answering/
def print_sources(k, query, source):
    num_sources = k
    source_retriever = loaded_vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": num_sources})
    source_results = source_retriever.get_relevant_documents(query)
    print("Sources for " + source + " were as follows:")
    for i in range(num_sources): print("\n\n--Source #" + str(i) + "\n" + source_results[i].page_content)
    print("\n\n")
# Alternatively
#### results = loaded_vectorStore.similarity_search_with_score(query) # returns 4 results by default
#### print(results)


# Method 1 - Prepare the Q&A bot - https://api.python.langchain.com/en/stable/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html?highlight=retrievalqawithsourceschain
# Provides only 1 input to the bot, and no control over the prompt
def method_1(query, print_source):
    llm = OpenAI(temperature=0,)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=loaded_vectorStore.as_retriever())
    result = chain({"question": query}, return_only_outputs=True)
    print("Output of method 1: " + result['answer'])
    # print("\n\n")
    if print_source:
        print_sources(1, query, "method_1")



# Method 2 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5
def method_2(query, print_source):
    llm = OpenAI(temperature=0,)
    prompt = hub.pull("rlm/rag-prompt")
    # Show the prompt if you want
    # print(
    #    prompt.invoke(
    #        {"context": "filler context", "question": "filler question"}
    #    ).to_string()
    #)
    num_sources = 4
    retriever = loaded_vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": num_sources})
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Output of method 2: ")
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
    print("\n\n")
    if print_source:
        print_sources(num_sources, query, "method_2")

    


# Method 3 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5 - custom prompt
def method_3(query, print_source):
    llm = OpenAI(temperature=0,)
    template = """You are answering questions based on the transcript of a presidential candidate interview. 
    Use the following pieces of context to answer the question at the end. 
    You should answer like a political commentator in the third person, and not as the candidate themselves.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Don't use knowledge outside of the provided context to try and enhance the answer.
    Use five sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)
    num_sources = 4
    retriever = loaded_vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": num_sources})

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

    output = rag_chain_with_source.invoke(query)
    print("Output of method 3: " + output['answer'])
    print("\n\n")
    if print_source:
        print_sources(num_sources, query, "method_3")



# Source for improvements - https://python.langchain.com/docs/use_cases/question_answering/
# Right now #2 has a superior answer to #3 

# Remaining steps
# 1. DONE - Try to improve the third prompt to get the best answer there, then have 3 iteratively more complex single q&a prompts
# 2. DONE - Set up the testing ground where it's easy to run multiple models together with the same query to see which is best
# 3. DONE - Create the simple UI for asking questions (try use that YT video from the aisan guy)
# 4. Will need to be able to ask multiple questions
# 5. Then update the embeddings on another candidate - will need to add some parsing on the name, or some means of maybe manually selecting (via) UI what candidate you're interested in
# 6. Anything else then on the readme sheet.