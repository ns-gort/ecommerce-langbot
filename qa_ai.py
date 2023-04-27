from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

class QABot:
    """
    A class to represent an AI Conversation bot

    ...

    Attributes
    ----------
    chat_history : list of str
        previous messages in chat
    qa: ConversationalRetrievalChain
        LLM Conversation Chain object -- what gives us chatbot functionality

    Methods
    -------
    query_answer(str):
        Reads a query and returns a chatbot answer
    clear_history():
        clears chatbot history
    """# 'ada' 'gpt-3.5-turbo' 'gpt-4',
    def __init__(self, vectorstore):
        
        # Set chat history
        
        self.chat_history = []

        # Define ConversationRetrievalChain using our vector store, use GPT-3.5-Turbo as our model 

        self.qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model='gpt-3.5-turbo'),retriever=vectorstore.as_retriever())

    
    def query_answer(self, query):
        """ 
        Respond to query using ConversationalRetrievalChain LLM model

        Keyword arguments:
        query -- the query to be answered
        
        returns:
        result -- response to user's query
        """
        
        # Use ConversationalRetrievalChain to generate answer to query
        
        result = self.qa({"question": query, "chat_history": self.chat_history})
        
        # Append response to chat history
        
        self.chat_history.append((query, result['answer']))
        
        # Return answer
        
        return result['answer']
    
    def clear_history(self):
        """
        Clears the bot's chat history
        """

        # Set chat_history to empty
        
        self.chat_history = []

        # Return None

        return 
