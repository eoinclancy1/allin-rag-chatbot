# Conversational Model Type 1
# Objective: Feed “context + history + query” into the same LLM and have that generate the response
# A direct copy of model_embeddings_prompt_testing_playground.py, but then it was since
# edited within model_embeddings_test_conv_ability3.py so that it could handle memory too
# This gets implemented in app.py 
import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain import hub, OpenAI
from langchain_core.runnables import RunnableParallel, RunnableLambda
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

def get_vectorstore(candiate_name):
    index = "embeddings/faiss_index_" + candiate_name
    return FAISS.load_local(index, embeddings)

# Just loading local embeddings
embeddings = OpenAIEmbeddings()
#loaded_vectorStore = FAISS.load_local("faiss_index_phillips", embeddings)

# Method 2 & 3 Source Printing
# Finding raw sources from the Vector database
#### Return up to k results from the vector source - https://python.langchain.com/docs/use_cases/question_answering/
def print_sources(k, query, source, loaded_vectorStore):
    num_sources = k
    source_retriever = loaded_vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": num_sources})
    source_results = source_retriever.get_relevant_documents(query)
    print("Sources for " + source + " were as follows:")
    for i in range(num_sources): print("\n\n--Source #" + str(i) + "\n" + source_results[i].page_content)
    print("\n\n")


# Method 3 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5 - custom prompt
def method_3(query, memory, candidate, print_source):
    llm = OpenAI(temperature=0,)

    template = """You are answering questions based on the transcript of a presidential candidate interview. 
    Use the following pieces of context to answer the question at the end. 
    You should answer like a political commentator in the third person, and not as the candidate themselves.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Don't use knowledge outside of the provided context to try and enhance the answer.
    Use five sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    
    However, in the case where the question seems to assume knowledge that happened earlier in the conversation, consult the last three conversations to see if there is relevant information there within the context of the conversation.
    Latest Conversation responses: {chat_history}
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)
    num_sources = 4
    retriever = get_vectorstore(candidate).as_retriever(search_type="similarity", search_kwargs={"k": num_sources})

    # this cookbook saved my life - https://python.langchain.com/docs/expression_language/cookbook/memory
    rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
            "chat_history": RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        } 
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever,
         "question": RunnablePassthrough(), 
         }
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }
    
    #if print_source:
    #    print_sources(num_sources, query, get_vectorstore(candidate), "method_3")
    output = rag_chain_with_source.invoke(query)
    memory.save_context({"input": query}, {"output": output['answer']})
    return {'answer': output['answer'], 'memory': memory}
    #return(output['answer'])
    


# Source for improvements - https://python.langchain.com/docs/use_cases/question_answering/
# Right now #2 has a superior answer to #3 

# Remaining steps
# 1. DONE - Try to improve the third prompt to get the best answer there, then have 3 iteratively more complex single q&a prompts
# 2. DONE - Set up the testing ground where it's easy to run multiple models together with the same query to see which is best
# 3. Create the simple UI for asking questions (try use that YT video from the aisan guy)
# 4. Will need to be able to ask multiple questions
# 5. Then update the embeddings on another candidate - will need to add some parsing on the name, or some means of maybe manually selecting (via) UI what candidate you're interested in
# 6. Anything else then on the readme sheet.