# improvement over the second attempt in script of same name. Uses a window buffer which should be better
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


# Method 3 via https://python.langchain.com/docs/use_cases/question_answering/ - see step 5 - custom prompt
def method_3(query, memory):
    llm = OpenAI(temperature=0,)
    
    template = """You are answering questions questions based on the transcript of a presidential candidate interview. 
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
ai_msg = method_3(question, memory)
print(ai_msg)
memory.save_context({"input": question}, {"output": ai_msg})
print(memory.load_memory_variables({}))



second_question = "What was my previous question?"
# The above question performs well but something like "What did I just ask about?" seems to refer to questions within the actual interview
ai_msg2 = method_3(second_question, memory)
print(ai_msg2)
memory.save_context({"input": second_question}, {"output": ai_msg2})
print(memory.load_memory_variables({}))
