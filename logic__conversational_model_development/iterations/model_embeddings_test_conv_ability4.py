# improvement over the third attempt in script of same name. 
# The objective is to use an earlier model which could generate an updated question
#     based on all the provided context, and provide that to the final RAG model. The
#     thought being that this would improve over 3 as some oddities occurred with the 3rd one
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

import os
from dotenv import load_dotenv
from operator import itemgetter


from langchain import hub, OpenAI
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
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
loaded_vectorStore = FAISS.load_local("faiss_index", embeddings)
llm = OpenAI(temperature=0,)

def condense_prompt(query, memory):
    
    condense_q_system_prompt = """Given a chat history and the latest user question
    which might reference the chat history, formulate a standalone question
    which can be understood without the chat history. 
    Where the subject of the question is unclear, prioritize the context in the "Human" provided 
    context over what is included in answers.
    
    Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    Under no circumstance should you answer your own question.
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
        )
    
    return(output)


# Method 3 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5 - custom prompt
def method_3(query, memory):
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
    
    output = rag_chain_with_source.invoke(query)
    return(output['answer'])



memory = ConversationBufferWindowMemory(k=3, return_messages=True)
memory.load_memory_variables({})



question = "What party is Dean Phillips associated with?"
#question_updated = condense_prompt(question, memory)
ai_msg = method_3(question, memory)
print(ai_msg)
memory.save_context({"input": question}, {"output": ai_msg})
#print(memory.load_memory_variables({}))
#print(memory.load_memory_variables({})['history'])



second_question = "Tell me something interesting about the candidate?"
# When I ask, 'what was his involvement with it?', it refers to the Problem Solvers Caucus instead of the 'party'.
second_question_updated = condense_prompt(second_question, memory)
print(second_question_updated)
ai_msg2 = method_3(second_question_updated, memory)
print(ai_msg2)
memory.save_context({"input": second_question_updated}, {"output": ai_msg2})
#print(memory.load_memory_variables({}))
