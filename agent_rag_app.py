import streamlit as st
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import AzureOpenAI, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import AzureSearch
from langchain.tools import Tool
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from azure.storage.blob import BlobServiceClient
from azure.cosmos import CosmosClient
import os
import json
from datetime import datetime
from typing import List, Dict

# Configure environment
class AzureConfig:
    def __init__(self):
        self.search_service_name = os.getenv("AZURE_SEARCH_SERVICE_NAME")
        self.search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.cosmos_endpoint = os.getenv("AZURE_COSMOS_ENDPOINT")
        self.cosmos_key = os.getenv("AZURE_COSMOS_KEY")
        self.storage_connection = os.getenv("AZURE_STORAGE_CONNECTION")

class MeetingAgent:
    def __init__(self, azure_config: AzureConfig):
        self.config = azure_config
        self.setup_azure_services()
        self.setup_agent()

    def setup_azure_services(self):
        """Initialize Azure services"""
        # Setup Azure OpenAI
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4",
            openai_api_version="2023-05-15",
            azure_endpoint=self.config.openai_endpoint,
            azure_deployment="gpt-4",
            api_key=self.config.openai_api_key
        )

        # Setup Azure Search
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-ada-002",
            openai_api_version="2023-05-15",
            azure_endpoint=self.config.openai_endpoint,
            api_key=self.config.openai_api_key
        )

        self.vector_store = AzureSearch(
            azure_search_endpoint=f"https://{self.config.search_service_name}.search.windows.net",
            azure_search_key=self.config.search_api_key,
            index_name="meeting-minutes-index",
            embedding_function=self.embeddings.embed_query
        )

    def setup_agent(self):
        """Setup the agent with tools and memory"""
        # Define tools
        search_tool = Tool(
            name="Search Meeting Minutes",
            func=self.vector_store.similarity_search,
            description="Search through meeting minutes for relevant information"
        )

        summarize_tool = Tool(
            name="Summarize Meeting",
            func=self.summarize_meeting,
            description="Generate a summary of meeting discussions"
        )

        extract_action_tool = Tool(
            name="Extract Actions",
            func=self.extract_action_items,
            description="Extract action items and assignments from the meeting"
        )

        tools = [search_tool, summarize_tool, extract_action_tool]

        # Setup memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create agent
        prompt = PromptTemplate.from_template("""
        You are an intelligent meeting assistant that helps users understand and analyze meeting minutes.
        Use the following tools to help answer questions and provide insights:
        
        {tools}
        
        Previous conversation context:
        {chat_history}
        
        Current question: {input}
        
        Think through this step by step:
        1. Understand what information is needed
        2. Decide which tool(s) to use
        3. Analyze the results
        4. Provide a clear, professional response
        
        Response:
        """)

        self.agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=tools,
            memory=self.memory,
            verbose=True
        )

    def summarize_meeting(self, text: str) -> str:
        """Generate a meeting summary"""
        prompt = """
        Analyze the following meeting minutes and provide a concise summary including:
        - Key discussion points
        - Decisions made
        - Next steps
        
        Meeting minutes:
        {text}
        """
        return self.llm.predict(prompt.format(text=text))

    def extract_action_items(self, text: str) -> List[Dict]:
        """Extract action items from meeting text"""
        prompt = """
        Extract action items from the following meeting minutes.
        For each action item, identify:
        - Task description
        - Assignee
        - Due date (if mentioned)
        - Priority (if mentioned)
        
        Return the results in a structured format.
        
        Meeting minutes:
        {text}
        """
        response = self.llm.predict(prompt.format(text=text))
        return json.loads(response)

def main():
    st.set_page_config(page_title="Meeting Minutes Assistant", layout="wide")
    
    # Initialize Azure configuration
    azure_config = AzureConfig()
    
    # Initialize agent
    if 'agent' not in st.session_state:
        st.session_state.agent = MeetingAgent(azure_config)
    
    st.title("Meeting Minutes Assistant")
    
    # File upload section
    with st.sidebar:
        st.header("Upload Meeting Minutes")
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'txt'])
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                # Process and store document
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(uploaded_file)
                else:
                    loader = TextLoader(uploaded_file)
                
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                
                # Store in vector store
                st.session_state.agent.vector_store.add_documents(texts)
                st.success("Document processed successfully!")
    
    # Chat interface
    st.header("Chat with your Meeting Minutes")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your meetings"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.agent_executor.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 
