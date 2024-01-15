# Conversational Model Type 2
# Objective: Chain 2 LLMs, with first obtaining “history + query” and objective to create an improved question” 
#            -> then feed “new question + context” into the 2nd LLM to generate the answer
# Pulled the logic from model_embeddings_test_conv_ability4.py, the main note
#    being that we pass all the context and a question to an earlier model that tries
#    to generate a better question, rather than let the rag do it all
# This gets implemented in app.py 
import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain import hub, OpenAI
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
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
#loaded_vectorStore = FAISS.load_local("faiss_index", embeddings)

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


def condense_prompt(query, memory):
    llm = OpenAI(temperature=0,)

    condense_q_system_prompt = """You are a question re-writer for the input to a chatbot.
    Given a chat history and the latest user question which might reference the chat history, 
    formulate a standalone question which can be understood without the chat history. 
    Where the subject of the question is unclear, prioritize the context in the "Human" provided 
    context over what is included in answers.
    You should always interpret the provided human input as a question, 
    that needs to be reviewed, and potentially rewritten given the provided context.
    
    STRICT RULES:
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    Under no circumstance should you answer your own question.
    Don't use knowledge outside of the provided context to try and enhance the answer.
    """

    condense_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", condense_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
    )
    condense_q_chain = condense_q_prompt | llm | StrOutputParser()

    output = condense_q_chain.invoke({
        "chat_history": memory.load_memory_variables({})['history'],
        "question": query,
        }
        ).strip('\n')
    
    if output.startswith('AI:'):
        return output[3:]
    
    return(output)


# Method 3 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5 - custom prompt
def method_3(query, memory, candidate, print_source):
    llm = OpenAI(temperature=0,)
    
    # Outside of updating this prompt to only return one sentence, it always seems to ramble a bit.
    template = """You are answering questions based on the transcript of a presidential candidate interview. 
    Use the following pieces of context to answer the question at the end. 
    You should answer like a political commentator in the third person, and not as the candidate themselves.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Don't use knowledge outside of the provided context to try and enhance the answer.
    Only answer the question asked, and brevity in your answer is preferred. 
    Use three sentences maximum and keep the answer as concise as possible.

    Context: 
    {context}
    
    Question: {question}
    
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)
    num_sources = 4
    retriever = get_vectorstore(candidate).as_retriever(search_type="similarity", search_kwargs={"k": num_sources})
    
    # this cookbook saved my life - https://python.langchain.com/docs/expression_language/cookbook/memory
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
    return(output['answer'])


def chained_prompt(query, memory, candidate, print_source):
    if memory.load_memory_variables({})['history']:
        revised_question = condense_prompt(query, memory)
        print('Revised Question: ' + revised_question)
        response = method_3(revised_question, memory, candidate, print_source)
        memory.save_context({"input": revised_question}, {"output": response})
        return {'answer': response, 'memory': memory}

    else:
        response = method_3(query, memory, candidate, print_source)
        memory.save_context({"input": query}, {"output": response})
        return {'answer': response, 'memory': memory}
         
    
