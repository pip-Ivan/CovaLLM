import os
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import json
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent parallelism warning

import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult, Document
from langchain.chains import LLMChain
from models import (
    BiodiversityCategory,
    Subcategory,
    CategoryScore,
    BiodiversityReport,
    BIODIVERSITY_FRAMEWORK,
    Question,
    analyze_text_content,
)
from agents import (
    BiodiversityAnalysisAgent,
    BiodiversityFrameworkAgent,
    search_project_forums_and_reviews,
    TokenUsageCallback,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from langchain_core.runnables import RunnableSequence
from langchain_community.vectorstores.faiss import FAISS

# Load environment variables
load_dotenv()


class ChunkStrategy:
    def __init__(self, pdf_chunk_size: int = 1000, tweet_chunk_size: int = 100):
        self.update_chunk_sizes(pdf_chunk_size, tweet_chunk_size)

    def update_chunk_sizes(self, pdf_chunk_size: int, tweet_chunk_size: int):
        """Update chunk sizes for both splitters"""
        self.pdf_splitter = RecursiveCharacterTextSplitter(
            chunk_size=pdf_chunk_size,
            chunk_overlap=int(pdf_chunk_size * 0.1),
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.tweet_splitter = RecursiveCharacterTextSplitter(
            chunk_size=tweet_chunk_size,
            chunk_overlap=0,
            length_function=len,
            separators=["\n", ".", " ", ""],
        )

        # Store sizes for reference
        self.pdf_chunk_size = pdf_chunk_size
        self.tweet_chunk_size = tweet_chunk_size

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents based on their source type"""
        pdf_docs = []
        tweet_docs = []
        other_docs = []

        # Separate documents by type
        for doc in documents:
            content_type = doc.metadata.get("content_type", "unknown")
            if content_type == "pdf":
                pdf_docs.append(doc)
            elif content_type == "tweet":
                tweet_docs.append(doc)
            else:
                other_docs.append(doc)

        # Process each type with appropriate splitter
        split_pdfs = self.pdf_splitter.split_documents(pdf_docs) if pdf_docs else []
        split_tweets = (
            self.tweet_splitter.split_documents(tweet_docs) if tweet_docs else []
        )
        split_others = (
            self.pdf_splitter.split_documents(other_docs) if other_docs else []
        )

        # Ensure content type is preserved in metadata
        for doc in split_pdfs:
            doc.metadata["content_type"] = "pdf"
        for doc in split_tweets:
            doc.metadata["content_type"] = "tweet"
        for doc in split_others:
            if "content_type" not in doc.metadata:
                doc.metadata["content_type"] = "unknown"

        print(
            f"Split into {len(split_pdfs)} PDF chunks, {len(split_tweets)} tweet chunks, and {len(split_others)} other chunks"
        )
        return split_pdfs + split_tweets + split_others

    @staticmethod
    def estimate_tokens(text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


class DataLoader:
    """Handles loading and preprocessing of different data sources"""
    
    @staticmethod
    def extract_field_with_fallbacks(row, field_options: List[str], default: str = "") -> str:
        """Extract field value using fallback field names"""
        for field in field_options:
            if field in row and row[field]:
                return str(row[field]).strip()
        return default

    @staticmethod
    def load_pdf(file_path: str) -> List[str]:
        """Load PDF with fallback mechanism, prioritizing UnstructuredPDFLoader"""
        try:
            print(f"Loading PDF: {file_path}")
            from langchain_community.document_loaders import (
                UnstructuredPDFLoader,
                PyPDFLoader,
            )

            # Try loaders in order of preference
            loaders = [
                ("UnstructuredPDFLoader", lambda: UnstructuredPDFLoader(file_path, strategy="fast")),
                ("PyPDFLoader", lambda: PyPDFLoader(file_path))
            ]
            
            for loader_name, loader_factory in loaders:
                try:
                    loader = loader_factory()
                    documents = loader.load()
                    if documents:
                        print(f"Successfully loaded PDF with {loader_name}: {file_path}")
                        return documents
                    else:
                        print(f"{loader_name} returned empty documents for {file_path}")
                except Exception as e:
                    print(f"Error with {loader_name} for {file_path}: {str(e)}")
                    continue
            
            print(f"All PDF loaders failed for {file_path}")
            return []
        except Exception as e:
            print(f"Critical error loading PDF {file_path}: {str(e)}")
            return []

    @staticmethod
    def load_excel(file_path: str) -> pd.DataFrame:
        try:
            df = pd.read_excel(file_path)
            return df
        except Exception as e:
            print(f"Error loading Excel {file_path}: {str(e)}")
            return pd.DataFrame()


class RAGAgent:
    # Model name mapping
    MODEL_MAPPING = {
        "gpt-4": "gpt-4",
        "gpt-4-turbo": "gpt-4-1106-preview",
        "gpt-3.5": "gpt-3.5-turbo",
        "gpt-4o-mini": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
    }

    def __init__(self, model_name: str, temperature: float = 0.1):
        self.model_name = self.MODEL_MAPPING.get(model_name)
        if not self.model_name:
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {list(self.MODEL_MAPPING.keys())}"
            )

        if self.model_name == "deepseek-chat":
            self.llm = ChatDeepSeek(
                model=self.model_name,
                temperature=temperature,
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )
        else:
            try:
                self.llm = ChatOpenAI(model=self.model_name, temperature=temperature)
            except Exception as e:
                raise ValueError(
                    f"Error initializing model {self.model_name}: {str(e)}"
                )

        self.callback_handler = TokenUsageCallback()
        self.vectorstore = None  # Will be set when needed

    def set_vectorstore(self, vectorstore):
        """Set the vector store for retrieval"""
        self.vectorstore = vectorstore

    def generate_response(
        self, query: str, vectorstore=None, k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a response to a user query using RAG.

        Args:
            query: The user's question
            vectorstore: The vector store to use for retrieval (optional)
            k: Number of documents to retrieve

        Returns:
            Dictionary containing the response text and additional data like plot information
        """
        if vectorstore:
            self.vectorstore = vectorstore

        if not self.vectorstore:
            return {"text": "No vector store available. Please select a project first."}

        # Check if this is a visualization request
        visualization_request = self._detect_visualization_request(query)

        # Check if this is a tweet-related query
        tweet_related = self._is_tweet_related_query(query)

        # Retrieve relevant documents
        try:
            # For tweet queries, increase k to find more relevant tweets
            search_k = k * 2 if tweet_related else k

            # Perform search with the appropriate k value
            relevant_docs = self.vectorstore.similarity_search(query, k=search_k)

            # Filter for tweet content if it's a tweet-related query
            if tweet_related:
                # Extract tweet documents
                tweet_docs = [
                    doc
                    for doc in relevant_docs
                    if doc.metadata.get("content_type") == "tweet"
                ]

                # If no tweets found, add a message to that effect
                if not tweet_docs:
                    return {
                        "text": "I couldn't find any tweet content related to your query. Please try another question or adjust your search terms."
                    }

                # Use the tweet documents for context and ensure we have at least some
                relevant_docs = tweet_docs[: min(k, len(tweet_docs))]

                if not relevant_docs:
                    return {
                        "text": "While I found some tweets, they don't seem to be directly relevant to your query. Please try rephrasing your question."
                    }

            # Format context from documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

            # Create prompt template based on request type
            if tweet_related:
                # Create a structured representation of tweets for the model
                tweet_context = ""
                for i, doc in enumerate(relevant_docs):
                    tweet_context += f"--- TWEET {i+1} ---\n"
                    tweet_context += doc.page_content + "\n"

                    # Add metadata if available
                    meta = doc.metadata
                    if meta.get("author_description"):
                        tweet_context += f"Author: {meta.get('author_description')}\n"
                    if meta.get("author_location"):
                        tweet_context += f"Location: {meta.get('author_location')}\n"
                    if meta.get("date"):
                        tweet_context += f"Date: {meta.get('date')}\n"

                    tweet_context += "---\n\n"

                template = """
                You are an assistant specialized in analyzing social media content related to biodiversity and environmental topics.
                
                Below are relevant tweets that match the user's query:
                
                {context}
                
                USER QUERY:
                {query}
                
                Respond to the query using ONLY the information from the tweets provided. 
                Include relevant author information, dates, and locations when pertinent.
                Make sure to cite specific information from the tweets to support your response.
                If you're asked about specific metrics or data that's not available in the tweets, politely indicate the limitation.
                
                Provide a clear and concise answer focused ONLY on what can be found in these tweets.
                
                RESPONSE:
                """
                # Replace the context with the structured tweet context
                context = tweet_context
            elif visualization_request:
                template = """
                You are an assistant specialized in biodiversity data analysis. 
                Use the following information to answer the question:
                
                CONTEXT:
                {context}
                
                USER QUERY:
                {query}
                
                The user is asking for a visualization. Please provide:
                1. A clear answer to their question
                2. Data for visualization in the following JSON format:
                
                For CATEGORICAL data (like species, impacts, stakeholders):
                ```json
                {{
                    "plot_type": "bar|pie",
                    "title": "Plot title here",
                    "x_label": "X-axis label (if applicable)",
                    "y_label": "Y-axis label (if applicable)",
                    "data_type": "categorical",
                    "data": {{
                        "categories": ["Category1", "Category2", ...],
                        "values": [value1, value2, ...],
                        "description": "Brief explanation of how values were assigned"
                    }}
                }}
                ```
                
                IMPORTANTLY, when dealing with categorical data that doesn't have explicit values:
                - For categories like impact types, species, stakeholders: AUTOMATICALLY ASSIGN frequency scores
                  (how often they're mentioned) or importance scores (based on context) on a scale of 1-5
                - For example, if "habitat loss" is mentioned more frequently or described as more severe than "pollution",
                  assign it a higher value (e.g., 5 vs 3)
                - Always explain your scoring logic in the "description" field
                
                If the data has TIME DIMENSION (trends, changes over time):
                ```json
                {{
                    "plot_type": "line",
                    "title": "Plot title here",
                    "x_label": "Time Period",
                    "y_label": "Value",
                    "data_type": "temporal",
                    "data": {{
                        "periods": ["Period1", "Period2", ...],
                        "values": [value1, value2, ...],
                        "description": "Explanation of the trend data"
                    }}
                }}
                ```
                
                ONLY include a plot if you can extract or reasonably infer quantitative relationships from the context.
                If no visualization is possible, include "visualization": null in your response.
                
                Your response should be formatted as:
                
                {{"answer": "Your detailed answer here. Make it clear and focused on the question.",
                 "visualization": ... visualization JSON here or null if not possible ...}}
                
                Structure your response exactly as shown above. Make sure to include both the answer and visualization fields.
                
                RESPONSE:
                """
            else:
                template = """
                You are an assistant specialized in biodiversity data analysis.
                Use the following information to answer the question:
                
                CONTEXT:
                {context}
                
                USER QUERY:
                {query}
                
                Provide a clear and concise answer based only on the information provided in the context.
                If you don't know the answer, say so clearly, don't make up information.
                
                RESPONSE:
                """

            # Generate response with the LLM
            with get_openai_callback() as cb:
                response = self.llm.invoke(
                    template.format(context=context, query=query),
                    config={"callbacks": [self.callback_handler]},
                ).content.strip()

            # Process the response
            if visualization_request:
                # Extract JSON data if present
                try:
                    # First, check if there's a JSON code block
                    json_match = re.search(
                        r"```json\s*(.*?)\s*```", response, re.DOTALL
                    )

                    result = {"text": response}

                    if json_match:
                        json_data = json_match.group(1)
                        try:
                            visualization_data = json.loads(json_data)

                            # Transform the visualization data to the expected format if needed
                            visualization_data = self._transform_visualization_data(
                                visualization_data
                            )

                            result["visualization"] = visualization_data
                        except json.JSONDecodeError:
                            # If the JSON is invalid, try to find the answer and vis keys
                            result["visualization_error"] = (
                                "Could not parse visualization data"
                            )
                    else:
                        # If no code block, try to parse the whole response as JSON
                        try:
                            # Try to find a complete JSON object
                            json_obj_match = re.search(r"(\{.*\})", response, re.DOTALL)
                            if json_obj_match:
                                json_str = json_obj_match.group(1)
                                response_obj = json.loads(json_str)

                                # Check if it has answer and visualization fields
                                if "answer" in response_obj:
                                    result["text"] = response_obj["answer"]

                                if "visualization" in response_obj:
                                    visualization_data = response_obj["visualization"]
                                    if visualization_data:
                                        # Transform the visualization data
                                        visualization_data = (
                                            self._transform_visualization_data(
                                                visualization_data
                                            )
                                        )
                                        result["visualization"] = visualization_data
                        except:
                            # If that fails, just return the original text
                            pass

                    return result
                except Exception as e:
                    print(f"Error extracting visualization data: {str(e)}")
                    return {"text": response, "visualization_error": str(e)}
            else:
                return {"text": response}

        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return {"text": f"Error generating response: {str(e)}"}

    def _transform_visualization_data(self, data: Dict) -> Dict:
        """
        Transform the visualization data to ensure compatibility with the display function.
        This function handles different data formats and ensures they're converted to the expected structure.
        """
        # Make a copy to avoid modifying the original
        transformed = data.copy()

        # Check if the data uses the new format with 'data_type' field
        if "data_type" in data and data["data_type"] == "categorical":
            # Convert from categorical format to the expected format
            if "data" in data:
                categories = data["data"].get("categories", [])
                values = data["data"].get("values", [])

                # Ensure the transformed data has the expected structure
                transformed["data"] = {"labels": categories, "values": values}

                # Remove the data_type field as it's not needed for display
                transformed.pop("data_type", None)

        elif "data_type" in data and data["data_type"] == "temporal":
            # Convert from temporal format to the expected format
            if "data" in data:
                periods = data["data"].get("periods", [])
                values = data["data"].get("values", [])

                # Ensure the transformed data has the expected structure
                transformed["data"] = {"labels": periods, "values": values}

                # Remove the data_type field as it's not needed for display
                transformed.pop("data_type", None)

        # Ensure we have plot_type if not specified
        if "plot_type" not in transformed:
            # Default to bar for categorical data
            transformed["plot_type"] = "bar"

        return transformed

    def _detect_visualization_request(self, query: str) -> bool:
        """
        Detect if the query is asking for a visualization

        Args:
            query: The user's question

        Returns:
            Boolean indicating whether visualization is requested
        """
        visualization_keywords = [
            "plot",
            "chart",
            "graph",
            "visualize",
            "visualise",
            "diagram",
            "show me",
            "display",
            "draw",
            "histogram",
            "pie chart",
            "bar chart",
            "trend",
            "distribution",
            "compare",
            "comparison",
            "breakdown",
            "analysis",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visualization_keywords)

    def _is_tweet_related_query(self, query: str) -> bool:
        """
        Determine if the query is asking about tweets or social media content.

        This function checks if the query contains keywords related to social media
        to determine if we should prioritize tweet documents in the search results.
        """
        query_lower = query.lower()

        # Define a more comprehensive set of keywords
        social_keywords = [
            "tweet",
            "twitter",
            "social media",
            "post",
            "hashtag",
            "follower",
            "user",
            "author",
            "social network",
            "sentiment",
            "opinion",
            "public reaction",
            "public opinion",
            "trending",
            "viral",
            "share",
            "retweet",
            "like",
            "comment",
            "thread",
            "timeline",
            "mentions",
            "social conversation",
            "discussion online",
            "what are people saying",
            "reactions",
            "location",
            "from where",
            "who is talking",
            "demographics",
            "public",
            "conversation",
            "response",
            "react",
            "said",
            "posted",
            "community",
            "audience",
            "engagement",
        ]

        # Check if any of the social keywords are in the query
        for keyword in social_keywords:
            if keyword in query_lower:
                print(f"Detected tweet-related query: '{query}' (matched '{keyword}')")
                return True

        # Also check for patterns like "what do tweets say about" or "how do people discuss"
        patterns = [
            r"what\s+do\s+(people|users|tweets?|posts?|authors?)\s+(say|mention|discuss)",
            r"how\s+(do|are)\s+(people|users|tweets?|posts?|authors?)\s+(talk|discuss|react)",
            r"(sentiment|tone|opinion|feeling)\s+of\s+(tweets?|posts?|people|users)",
            r"who\s+(is|are)\s+(talk|post|tweet|discuss|mention)ing",
            r"where\s+(is|are)\s+(people|users)\s+(talk|post|tweet)",
            r"when\s+(did|was|were)\s+(people|users)\s+(talk|post|tweet)",
        ]

        for pattern in patterns:
            if re.search(pattern, query_lower):
                print(f"Detected tweet-related query via pattern matching: '{query}'")
                return True

        return False

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def analyze_text(self, question: str, context: str) -> Dict[str, Any]:
        """
        Analyze text based on a specific question using the LLM.

        Args:
            question: The specific question to answer
            context: The context to use for analysis

        Returns:
            Dictionary with analysis results
        """
        # Check if this is a social media/tweet-related question
        is_social_media = self._is_tweet_related_query(question)

        # Custom prompt for different types of analysis
        if is_social_media:
            prompt = PromptTemplate(
                template="""You are an expert at analyzing social media content.
                
                CONTEXT:
                {context}
                
                QUESTION:
                {question}
                
                Provide a detailed analysis of the social media content related to the question.
                Extract relevant trends, sentiments, and key topics from the social media content.
                
                In your analysis, include:
                1. Key trends and topics from the social media content
                2. Overall sentiment (positive, negative, neutral, mixed)
                3. Notable quotes or representative opinions
                4. Common hashtags or themes
                5. Metadata insights (locations, author types, date ranges if available)
                
                Format your response as JSON with these keys:
                {{"analysis": "Your textual analysis here",
                "sentiment": "overall sentiment (positive/negative/neutral/mixed)",
                "key_topics": ["topic1", "topic2", ...],
                "notable_quotes": ["quote1", "quote2", ...],
                "hashtags": ["hashtag1", "hashtag2", ...] or [],
                "metadata_insights": {{"locations": ["location1", "location2", ...] or [],
                                     "author_types": ["type1", "type2", ...] or [],
                                     "date_range": "description of date range" or null}}
                }}
                
                IMPORTANT: Make sure your output is a valid JSON object.
                """,
                input_variables=["context", "question"],
            )
        else:
            # Original prompt for non-social media content
            prompt = PromptTemplate(
                template="""As an environmentalist and biodiversity expert, answer the following question based on the provided information.
                
                CONTEXT INFORMATION:
                {context}
                
                QUESTION:
                {question}
                
                Provide a thorough answer that:
                1. Directly addresses the question
                2. Focuses on biodiversity aspects
                3. Incorporates specific data from the context
                4. Notes any potential limitations in the available information
                
                Your answer:""",
                input_variables=["context", "question"],
            )

        callbacks = [self.callback_handler]
        response = ""
        try:
            chain = LLMChain(prompt=prompt, llm=self.llm)
            response = chain.run(
                context=context, question=question, callbacks=callbacks
            )
        except Exception as e:
            response = f"Error in analysis: {str(e)}"

        result = {"text": response}

        # For social media content, try to parse the JSON
        if is_social_media and response:
            try:
                # Extract JSON from the response if it's not already parsed
                if not response.startswith("{"):
                    # Try to find JSON in the response
                    json_match = re.search(r"({.*})", response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        result["social_media_data"] = json.loads(json_str)
                else:
                    result["social_media_data"] = json.loads(response)

                # Make sure we always return the text for compatibility
                if "text" not in result:
                    result["text"] = result["social_media_data"].get(
                        "analysis", response
                    )
            except Exception as e:
                print(f"Error parsing social media JSON response: {str(e)}")
                # Leave the original text response intact

        return result

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_tokens": self.callback_handler.total_tokens,
            "total_cost": self.callback_handler.total_cost,
            "successful_requests": self.callback_handler.successful_requests,
            "failed_requests": self.callback_handler.failed_requests,
        }


class RAGSystem:
    def __init__(self, base_directory: str = None, model_name: str = "gpt-4o-mini"):
        self.base_directory = base_directory or os.getenv("BASE_DIRECTORY")
        if not self.base_directory:
            raise ValueError("Base directory not specified")

        # Create vector store directory if it doesn't exist
        self.vector_store_dir = os.path.join(self.base_directory, ".vector_stores")
        os.makedirs(self.vector_store_dir, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"  # More cost-effective than text-embedding-3-large
        )

        # Initialize chunk strategy with default sizes
        self.chunk_strategy = ChunkStrategy(pdf_chunk_size=1000, tweet_chunk_size=100)

        # Initialize with the selected RAG agent
        self.model_name = model_name
        default_agent = RAGAgent(model_name)
        self.llm = default_agent.llm

        # Initialize specialized agents with the selected LLM - streamlined with combined analysis
        self.agents = {
            "primary": default_agent,  # Main query agent
            "biodiversity": BiodiversityAnalysisAgent(
                self.llm
            ),  # Combined sentiment and social media analysis
            "framework": BiodiversityFrameworkAgent(
                self.llm
            ),  # Core framework evaluation agent
        }

        # Forum and review search functionality available via search_project_forums_and_reviews function

        self.vectorstore = None
        self.current_project = None
        self.social_data = None
        self.biodiversity_report = None

    def update_model(self, model_name: str):
        """Update the LLM model for all agents"""
        if model_name != self.model_name:
            print(f"Updating model from {self.model_name} to {model_name}")
            self.model_name = model_name
            
            # Create new agent with the selected model
            new_agent = RAGAgent(model_name)
            self.llm = new_agent.llm
            
            # Update all agents with the new LLM
            self.agents["primary"] = new_agent
            self.agents["biodiversity"] = BiodiversityAnalysisAgent(self.llm)
            self.agents["framework"] = BiodiversityFrameworkAgent(self.llm)
            
            # Reconnect the primary agent to vectorstore if available
            if self.vectorstore:
                self.agents["primary"].set_vectorstore(self.vectorstore)

    def get_vectorstore(self):
        """Get the current vectorstore for use in other components"""
        return self.vectorstore

    def connect_agent_to_vectorstore(self, agent: RAGAgent):
        """Connect a RAG agent to the current vectorstore"""
        if self.vectorstore:
            agent.set_vectorstore(self.vectorstore)
            return True
        return False

    def get_available_projects(self) -> List[str]:
        """Get list of available project directories."""
        return [
            d
            for d in os.listdir(self.base_directory)
            if os.path.isdir(os.path.join(self.base_directory, d))
            and not d.startswith(".")
        ]

    def set_current_project(self, project_name: str) -> None:
        """Set the current project and load its data"""
        project_path = os.path.join(self.base_directory, project_name)
        if not os.path.exists(project_path):
            raise ValueError(f"Project directory not found: {project_path}")

        self.current_project = project_name

        # Load social media data if available
        social_data_path = os.path.join(project_path, "social_data.xlsx")
        if os.path.exists(social_data_path):
            self.social_data = DataLoader.load_excel(social_data_path)
            print(f"Loaded social media data: {len(self.social_data)} records")
        else:
            print("No social media data found")
            self.social_data = None

        # Fetch forum data based on project name
        try:
            print(f"Fetching forum discussions and reviews for project: {project_name}")
            self.forum_context = search_project_forums_and_reviews(project_name, max_results=8)
            if self.forum_context and "No forum discussions or reviews found" not in self.forum_context:
                print("Found relevant forum discussions and reviews")
            else:
                print("No relevant forum discussions or reviews found")
                self.forum_context = None
        except Exception as e:
            print(f"Error fetching forum data: {str(e)}")
            self.forum_context = None

        # Load or create vector store
        vector_store_path = os.path.join(self.vector_store_dir, f"{project_name}_store")
        try:
            if os.path.exists(vector_store_path):
                print(f"Loading existing vector store for project {project_name}...")
                self.vectorstore = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )

                # Verify content types in loaded vector store
                test_docs = self.vectorstore.similarity_search("test", k=10)
                pdf_count = sum(
                    1 for doc in test_docs if doc.metadata.get("content_type") == "pdf"
                )
                tweet_count = sum(
                    1
                    for doc in test_docs
                    if doc.metadata.get("content_type") == "tweet"
                )
                print(
                    f"Vector store loaded with {pdf_count} PDFs and {tweet_count} tweets in sample"
                )

                # If no PDFs found in sample, try a more targeted search
                if pdf_count == 0 and test_docs:
                    # Try a more specific search that might target PDF content
                    pdf_test_docs = self.vectorstore.similarity_search(
                        "report document biodiversity", k=10
                    )
                    pdf_count = sum(
                        1
                        for doc in pdf_test_docs
                        if doc.metadata.get("content_type") == "pdf"
                    )
                    print(f"Secondary check: found {pdf_count} PDFs in targeted search")

                # If still no content types found or no PDFs, we need to rebuild the vector store
                if (pdf_count == 0 and tweet_count == 0 and test_docs) or (
                    pdf_count == 0 and test_docs
                ):
                    print(
                        "Warning: No PDF content found in vector store. Rebuilding..."
                    )
                    self.process_project_data(project_path, vector_store_path)
                else:
                    print("Vector store loaded successfully")

                    # Connect the primary agent to the vectorstore
                    if "primary" in self.agents:
                        self.agents["primary"].set_vectorstore(self.vectorstore)

                    return
            else:
                print("No existing vector store found. Creating new one...")
                self.process_project_data(project_path, vector_store_path)
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            print("Creating new vector store...")
            self.process_project_data(project_path, vector_store_path)

        # Connect the primary agent to the vectorstore after processing
        if "primary" in self.agents:
            self.agents["primary"].set_vectorstore(self.vectorstore)

    def process_project_data(self, project_path: str, vector_store_path: str) -> None:
        """Process all project data including PDFs and social media"""
        documents = []

        # Process PDFs - check for both uppercase and lowercase extensions
        pdf_files = [
            f
            for f in os.listdir(project_path)
            if f.lower().endswith((".pdf", ".PDF"))
            and os.path.isfile(os.path.join(project_path, f))
        ]

        print(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")

        # Additional verification of PDF files
        verified_pdf_files = []
        for pdf_file in pdf_files:
            full_path = os.path.join(project_path, pdf_file)
            file_size = os.path.getsize(full_path)
            if file_size > 0:
                verified_pdf_files.append(pdf_file)
                print(f"Verified PDF file: {pdf_file}, size: {file_size} bytes")
            else:
                print(f"Skipping empty PDF file: {pdf_file}")

        pdf_files = verified_pdf_files

        # Process PDFs in parallel with better resource management
        max_workers = min(4, len(pdf_files) or 1, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all PDF processing tasks
            pdf_futures = {
                executor.submit(DataLoader.load_pdf, os.path.join(project_path, pdf_file)): pdf_file 
                for pdf_file in pdf_files
            }

            # Process results as they complete
            for future in tqdm(pdf_futures, total=len(pdf_files), desc="Processing PDFs"):
                pdf_file = pdf_futures[future]
                try:
                    docs = future.result()
                    # Add content type to metadata for each document
                    for doc in docs:
                        if not hasattr(doc, "metadata"):
                            doc.metadata = {}
                        doc.metadata.update({
                            "content_type": "pdf",
                            "source": "pdf",
                            "filename": pdf_file
                        })
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error processing PDF {pdf_file}: {str(e)}")

        # Process social media data if available
        if self.social_data is not None:
            print(f"Processing {len(self.social_data)} social media records")
            # Convert relevant social media content to documents
            social_docs = self._process_social_data()
            documents.extend(social_docs)

        if not documents:
            print("Warning: No documents were successfully processed")
            # Create a dummy document to prevent errors
            documents = [
                Document(
                    page_content="No content available for analysis.",
                    metadata={"content_type": "none", "source": "none"},
                )
            ]

        # Create vector store
        try:
            print("Creating vector store...")
            print(f"Total documents before splitting: {len(documents)}")

            # Verify all documents have metadata
            for i, doc in enumerate(documents):
                if not hasattr(doc, "metadata"):
                    print(
                        f"Warning: Document {i} has no metadata. Adding default metadata."
                    )
                    doc.metadata = {"content_type": "unknown", "source": "unknown"}
                elif not isinstance(doc.metadata, dict):
                    print(
                        f"Warning: Document {i} has invalid metadata type: {type(doc.metadata)}. Fixing."
                    )
                    doc.metadata = {"content_type": "unknown", "source": "unknown"}

            # Split documents
            splits = self.chunk_strategy.split_documents(documents)
            print(f"Total chunks after splitting: {len(splits)}")

            # Verify content types after splitting
            pdf_chunks = sum(
                1 for doc in splits if doc.metadata.get("content_type") == "pdf"
            )
            tweet_chunks = sum(
                1 for doc in splits if doc.metadata.get("content_type") == "tweet"
            )
            print(f"- PDF chunks: {pdf_chunks}")
            print(f"- Tweet chunks: {tweet_chunks}")

            # Create vector store
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)
            self.vectorstore.save_local(vector_store_path)
            print("Vector store created and saved successfully")
        except Exception as e:
            print(f"Error during document processing: {str(e)}")
            import traceback

            traceback.print_exc()
            raise ValueError(f"Error during document processing: {str(e)}")

    def _process_social_data(self) -> List[Document]:
        """Convert social media data to documents for vector store"""
        if self.social_data is None:
            return []

        # Debug: Print the columns available in the dataset
        print(
            f"Available columns in social media data: {list(self.social_data.columns)}"
        )

        # Debug: Print sample data
        print(f"Sample data (first row):")
        try:
            sample_row = self.social_data.iloc[0]
            for col in self.social_data.columns:
                print(f"  {col}: {sample_row.get(col, 'N/A')}")
        except Exception as e:
            print(f"Error printing sample: {str(e)}")

        def preprocess_tweet(text: str) -> str:
            """Clean and preprocess tweet text"""
            if not isinstance(text, str):
                return ""
            # Remove URLs
            text = " ".join(
                word for word in text.split() if not word.startswith("http")
            )
            # Remove multiple spaces
            text = " ".join(text.split())
            return text.strip()

        # Collect stats for analytics
        author_occupations = set()
        locations = set()
        dates = set()
        empty_tweets = 0
        processed_tweets = 0

        documents = []
        for _, row in self.social_data.iterrows():
            # Use utility function for tweet text extraction
            tweet_text = DataLoader.extract_field_with_fallbacks(
                row, ["fullText", "text", "content", "tweet", "description", "summary"]
            )
            if tweet_text:
                tweet_text = preprocess_tweet(tweet_text)

            # If still no text, skip this row
            if not tweet_text:
                empty_tweets += 1
                continue

            # Extract additional fields using utility function
            # Use utility function for author description
            author_desc = DataLoader.extract_field_with_fallbacks(
                row, [
                    "author/description",
                    "authorDescription", 
                    "user_description",
                    "userDescription",
                ]
            )
            if author_desc:
                author_occupations.add(author_desc)

            # Use utility function for location
            author_location = DataLoader.extract_field_with_fallbacks(
                row, [
                    "author/location",
                    "authorLocation",
                    "user_location",
                    "userLocation",
                    "location",
                ]
            )
            if author_location:
                locations.add(author_location)

            # Use utility function for date
            created_at = DataLoader.extract_field_with_fallbacks(
                row, [
                    "author/createdAt",
                    "createdAt",
                    "created_at", 
                    "date",
                    "timestamp",
                ]
            )

            # Format date if it exists
            date_str = ""
            if created_at:
                try:
                    # Attempt to parse and format the date
                    if isinstance(created_at, str):
                        date_str = created_at
                    elif pd.notnull(created_at):
                        date_str = str(created_at)

                    if date_str:
                        dates.add(date_str)
                except Exception as e:
                    print(f"Error formatting date {created_at}: {str(e)}")
                    if pd.notnull(created_at):
                        date_str = str(created_at)

            # Use efficient string building with list and join
            text_parts = [f"Tweet Content: {tweet_text}"]

            # Add author information if available
            if author_desc:
                text_parts.append(f"Author Description: {author_desc}")
            if author_location:
                text_parts.append(f"Author Location: {author_location}")
            if date_str:
                text_parts.append(f"Created At: {date_str}")

            # Add hashtags using utility function
            hashtags = DataLoader.extract_field_with_fallbacks(row, ["hashtags", "tags"])
            if hashtags:
                if isinstance(hashtags, str):
                    text_parts.append(f"Hashtags: {hashtags}")
                elif isinstance(hashtags, list):
                    text_parts.append(f"Hashtags: {', '.join(hashtags)}")

            # Try to extract engagement metrics using utility function
            engagement = 0
            engagement_raw = DataLoader.extract_field_with_fallbacks(
                row, ["engagement", "likes", "retweets", "favorites"]
            )
            if engagement_raw:
                try:
                    engagement = float(engagement_raw)
                    text_parts.append(f"Engagement Metrics: {engagement}")
                except (ValueError, TypeError):
                    pass

            # Efficiently combine all text parts
            text = "\n".join(text_parts)

            doc = Document(
                page_content=text,
                metadata={
                    "source": "social_media",
                    "content_type": "tweet",  # Explicitly set content type
                    "date": date_str,
                    "author_description": author_desc,
                    "author_location": author_location,
                    "engagement": float(engagement),
                    "type": "tweet",
                    "raw_text": tweet_text,  # Store original text for reference
                },
            )
            documents.append(doc)
            processed_tweets += 1

        print(f"Processed {len(documents)} valid tweets with enhanced metadata")
        if empty_tweets > 0:
            print(f"Warning: {empty_tweets} tweets were skipped due to empty content")

        # Store the stats
        self.social_media_stats = {
            "total_tweets": processed_tweets + empty_tweets,
            "processed_tweets": processed_tweets,
            "empty_tweets": empty_tweets,
            "unique_locations": list(locations),
            "unique_author_types": list(author_occupations),
            "date_range": {
                "earliest": min(dates) if dates else None,
                "latest": max(dates) if dates else None,
            },
        }

        print(
            f"Social media stats: {len(locations)} locations, {len(author_occupations)} author types"
        )
        return documents

    def get_social_media_stats(self) -> Dict[str, Any]:
        """Return statistics about the social media data"""
        if not hasattr(self, "social_media_stats"):
            return {
                "total_tweets": 0,
                "processed_tweets": 0,
                "empty_tweets": 0,
                "unique_locations": [],
                "unique_author_types": [],
                "date_range": {"earliest": None, "latest": None},
            }
        return self.social_media_stats

    def _calculate_specificity(self, text: str) -> float:
        """
        Calculate a specificity score (0-100) for the given text.
        Specificity measures how detailed and actionable the content is.
        Higher score means more specific, detailed information with clear action plans
        and supporting data.
        """
        try:
            # Use semantic patterns to identify specificity indicators
            indicators = {
                # Numeric data and statistics indicate specificity
                "measurements": len(
                    re.findall(
                        r"\b\d+\s*(?:km|m|cm|mm|kg|g|mg|tons?|hectares?|acres?|%|percent)\b",
                        text,
                    )
                ),
                "statistics": len(
                    re.findall(r"\d+(?:\.\d+)?%|\d+\s*percent|\d+\s*of\s*\d+", text)
                ),
                # Action plans and implementation details
                "action_verbs": len(
                    re.findall(
                        r"\b(?:implement|execute|conduct|perform|establish|create|develop|launch|initiate|introduce|prepare|provide)\b",
                        text.lower(),
                    )
                ),
                "timeline": len(
                    re.findall(
                        r"\b(?:by|in|within)\s+\d+\s+(?:days?|weeks?|months?|years?|quarters?)\b",
                        text.lower(),
                    )
                ),
                # Methodology and process descriptions
                "methodologies": len(
                    re.findall(
                        r"\b(?:method|approach|process|procedure|technique|strategy|framework|protocol|guideline)\b",
                        text.lower(),
                    )
                ),
                # Quantifiable goals and targets
                "targets": len(
                    re.findall(
                        r"\b(?:target|goal|objective|aim)\s+(?:of|to|is|are)\s+\d+",
                        text.lower(),
                    )
                ),
                # Specificity in stakeholder mentions
                "named_entities": len(
                    re.findall(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)+\b", text)
                ),
                # Detailed planning terms
                "planning_terms": len(
                    re.findall(
                        r"\b(?:plan|schedule|roadmap|blueprint|outline|scheme|program|arrangement|design)\b",
                        text.lower(),
                    )
                ),
            }

            # Calculate base score based on indicators
            # Each category can contribute up to 12.5 points (for a total of 100)
            specificity_score = 0

            for key, count in indicators.items():
                if key in ["measurements", "statistics"]:
                    # Data support is highly valuable for specificity
                    specificity_score += min(15, count * 3)
                elif key in ["action_verbs", "timeline"]:
                    # Action plans are critical for specificity
                    specificity_score += min(15, count * 2.5)
                elif key in ["methodologies", "targets"]:
                    # Methodologies and targets show detailed planning
                    specificity_score += min(12.5, count * 2)
                else:
                    # Other indicators
                    specificity_score += min(10, count * 1.5)

            # Look for detailed paragraph structure (long complex sentences tend to have more detail)
            sentences = re.split(r"[.!?]+", text)
            avg_sentence_length = sum(
                len(s.split()) for s in sentences if s.strip()
            ) / max(1, len([s for s in sentences if s.strip()]))

            # Longer average sentences can indicate more detailed content
            if avg_sentence_length > 15:
                specificity_score += 10
            elif avg_sentence_length > 10:
                specificity_score += 5

            # Check for presence of bullet points or numbered lists (indicates structured information)
            if re.search(r"(?:^|\n)\s*[\*\-•]\s+", text) or re.search(
                r"(?:^|\n)\s*\d+\.\s+", text
            ):
                specificity_score += 10

            # Cap at 100
            return min(100.0, specificity_score)
        except Exception as e:
            print(f"Error calculating specificity: {e}")
            return 50.0  # Default to middle value

    def _calculate_multiplicity(self, relevant_excerpts: List[str]) -> float:
        """
        Calculate the redundancy/multiplicity score (0-100).
        This measures what percentage of mentions are repeated across excerpts.
        Higher score means more redundant information.

        Parameters:
            relevant_excerpts: List of text excerpts considered relevant

        Returns:
            A score from 0-100 representing the percentage of redundant mentions
        """
        if not relevant_excerpts or len(relevant_excerpts) <= 1:
            return 0.0  # No redundancy with 0 or 1 excerpt

        try:
            # Extract key concepts from each excerpt
            concept_mentions = []

            # First, identify key concepts in each excerpt
            for excerpt in relevant_excerpts:
                # Extract named entities (likely important concepts)
                entities = re.findall(r"\b[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b", excerpt)

                # Extract key noun phrases and technical terms
                noun_phrases = re.findall(
                    r"\b(?:[A-Za-z]+\s+){1,3}(?:plan|strategy|system|program|project|initiative|measure|policy|protocol|framework|method|approach)\b",
                    excerpt.lower(),
                )

                # Extract biodiversity-specific terms
                biodiversity_terms = re.findall(
                    r"\b(?:species|habitat|ecosystem|conservation|biodiversity|wildlife|flora|fauna|endangered|threatened|protected|reserve|sustainable|preservation)\b",
                    excerpt.lower(),
                )

                # Combine all detected concepts
                concepts = entities + noun_phrases + biodiversity_terms

                # Add concepts to our tracking list
                concept_mentions.extend([concept.lower() for concept in concepts])

            # If no concepts found, try a simpler approach with important phrases
            if not concept_mentions:
                for excerpt in relevant_excerpts:
                    # Look for noun-verb combinations (likely key actions)
                    phrases = re.findall(
                        r"\b(?:[a-z]+\s+){1,2}(?:is|are|will|should|must|can|could|would|has|have|had)\b",
                        excerpt.lower(),
                    )
                    concept_mentions.extend(phrases)

            # Count total mentions and redundant mentions
            if not concept_mentions:
                return 10.0  # Default low redundancy if no clear concepts found

            total_mentions = len(concept_mentions)
            unique_concepts = set(concept_mentions)
            unique_mentions = len(unique_concepts)

            # Calculate redundancy as percentage of duplicate mentions
            if unique_mentions == 0:
                return 0.0

            redundant_mentions = total_mentions - unique_mentions
            redundancy_percentage = (redundant_mentions / total_mentions) * 100

            return min(100.0, redundancy_percentage)
        except Exception as e:
            print(f"Error calculating multiplicity: {e}")
            return 10.0  # Default to low redundancy

    def evaluate_project(self, framework=None) -> BiodiversityReport:
        """
        Evaluate the current project against the biodiversity framework.
        """
        if framework is None:
            framework = BIODIVERSITY_FRAMEWORK

        # Initialize the biodiversity report with categories
        self.biodiversity_report = BiodiversityReport(category_scores=framework)

        # Compile all text content for analysis
        text_content = ""

        # Get text from documents in vectorstore if available
        if self.vectorstore:
            # Sample documents to get representative content
            sample_docs = self.vectorstore.similarity_search(
                "biodiversity conservation stakeholder", k=20
            )
            if sample_docs:
                text_content = "\n".join([doc.page_content for doc in sample_docs])

        # Use the analyze_text_content function to process the report
        if text_content:
            self.biodiversity_report = analyze_text_content(
                text_content, self.biodiversity_report, self
            )
        else:
            print("No text content available for analysis. Analysis may be incomplete.")
            # Still use analyze_text_content but with an empty string
            self.biodiversity_report = analyze_text_content(
                "", self.biodiversity_report, self
            )

        # Normalize insights for the final report
        self.biodiversity_report.normalize_insights()

        return self.biodiversity_report

    def get_token_stats(self) -> Dict[str, Dict[str, Any]]:
        return {name: agent.get_stats() for name, agent in self.agents.items()}
