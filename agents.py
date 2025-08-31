from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.schema import LLMResult
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()




class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.successful_requests = 0
        self.failed_requests = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        if hasattr(response, "llm_output"):
            if "token_usage" in response.llm_output:
                usage = response.llm_output["token_usage"]
                self.total_tokens += usage.get("total_tokens", 0)
                self.total_cost += self._calculate_cost(usage)
                self.successful_requests += 1

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.failed_requests += 1

    def _calculate_cost(self, usage: dict) -> float:
        # Cost per 1K tokens (approximate rates)
        rates = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4o-mini": {"input": 0.002, "output": 0.003},
            "deepseek-chat": {"input": 0.002, "output": 0.002},
        }
        model = usage.get("model", "gpt-4o-mini")
        rate = rates.get(model, rates["gpt-4o-mini"])
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        return (
            prompt_tokens * rate["input"] + completion_tokens * rate["output"]
        ) / 1000


class AnalysisAgent(ABC):
    """Base class for specialized analysis agents"""

    def __init__(self, llm):
        self.llm = llm
        self.callback_handler = TokenUsageCallback()
        # Define the output format for all agents
        self.expected_parts = 1  # Default, should be overridden by subclasses

    @abstractmethod
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the provided context and return results"""
        pass

    def _create_prompt_template(self) -> PromptTemplate:
        """Create a prompt template - should be implemented by subclasses"""
        raise NotImplementedError

    def _process_response(self, response: str) -> Dict[str, Any]:
        """Process the response - should be implemented by subclasses"""
        raise NotImplementedError

    def _get_default_results(self) -> Dict[str, Any]:
        """Get default results for when analysis fails"""
        raise NotImplementedError

    def _safe_split(self, response: str, expected_parts: int = None) -> List[str]:
        """Safely split response and ensure correct number of parts"""
        if expected_parts is None:
            expected_parts = self.expected_parts

        # First, clean the response
        response = response.strip()

        # Check if response is empty or doesn't contain separator
        if not response or "|" not in response:
            print(f"Invalid response format (no separators): {response}")
            return ["0"] + [""] * (expected_parts - 1)

        # Split the response
        parts = response.split("|")

        if len(parts) > expected_parts:
            print(
                f"Warning: Expected {expected_parts} parts, got {len(parts)}. Using first {expected_parts} parts."
            )
            print(f"Full response was: {response}")
            return parts[:expected_parts]
        elif len(parts) < expected_parts:
            print(
                f"Warning: Expected {expected_parts} parts, got {len(parts)}. Padding with empty values."
            )
            print(f"Full response was: {response}")
            return parts + [""] * (expected_parts - len(parts))

        return parts

    def _validate_response(self, response: str) -> bool:
        """Validate the response format"""
        if not response or not isinstance(response, str):
            return False
        if "|" not in response:
            return False
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return {
            "total_tokens": self.callback_handler.total_tokens,
            "total_cost": self.callback_handler.total_cost,
            "successful_requests": self.callback_handler.successful_requests,
            "failed_requests": self.callback_handler.failed_requests,
        }


class BiodiversityAnalysisAgent(AnalysisAgent):
    """Combined agent for biodiversity sentiment and social media analysis"""

    def __init__(self, llm):
        super().__init__(llm)
        self.expected_parts = 6  # Enhanced to handle both sentiment and social data

    def _create_prompt_template(self) -> PromptTemplate:
        template = """
        Analyze the following content for biodiversity and environmental topics. The content may include social media posts, forum discussions, or project documents:
        
        {text}
        
        INSTRUCTIONS:
        - Analyze overall sentiment regarding biodiversity/environmental initiatives
        - Identify key biodiversity topics and themes
        - Assess engagement level and community response
        - Extract positive and negative aspects
        - Identify key emotions present
        - Consider author expertise, locations, and temporal context if available
        
        YOU MUST RESPOND IN EXACTLY THIS FORMAT WITH 6 PARTS SEPARATED BY |:
        <sentiment_score>|<engagement>|<topics>|<positive>|<negative>|<emotions>

        WHERE:
        - <sentiment_score>: A single number between 0 and 1 (0=negative, 0.5=neutral, 1=positive)
        - <engagement>: A single number between 0 and 1 (biodiversity relevance and specificity)
        - <topics>: Comma-separated biodiversity/environment topics (no spaces after commas)
        - <positive>: Comma-separated positive aspects (no spaces after commas)
        - <negative>: Comma-separated negative aspects (no spaces after commas)
        - <emotions>: Comma-separated emotions (no spaces after commas)

        EXAMPLE RESPONSE:
        0.75|0.8|biodiversity,conservation,species|strong commitment,clear goals|lack of funding,timeline unclear|optimistic,concerned

        RESPOND WITH ONLY THE FORMATTED TEXT. NO OTHER TEXT OR EXPLANATION.
        """
        return PromptTemplate(template=template, input_variables=["text"])

    def _get_default_results(self) -> Dict[str, Any]:
        return {
            "sentiment_score": 0.5,
            "engagement_score": 0.5,
            "key_topics": [],
            "positive_aspects": [],
            "negative_aspects": [],
            "key_emotions": [],
            "sentiment_distribution": {"neutral": 100},
            "community_response": "neutral",
        }

    def _process_response(self, response: str) -> Dict[str, Any]:
        parts = self._safe_split(response, 6)
        
        try:
            sentiment_score = float(parts[0])
            if not 0 <= sentiment_score <= 1:
                sentiment_score = 0.5
        except (ValueError, IndexError):
            sentiment_score = 0.5

        try:
            engagement_score = float(parts[1])
            if not 0 <= engagement_score <= 1:
                engagement_score = 0.5
        except (ValueError, IndexError):
            engagement_score = 0.5

        # Extract topics, positive, negative, emotions
        key_topics = [x.strip() for x in parts[2].split(",") if x.strip()]
        positive_aspects = [x.strip() for x in parts[3].split(",") if x.strip()]
        negative_aspects = [x.strip() for x in parts[4].split(",") if x.strip()]
        key_emotions = [x.strip() for x in parts[5].split(",") if x.strip()]

        # Create sentiment distribution based on sentiment score
        if sentiment_score >= 0.7:
            sentiment_distribution = {"positive": 70, "neutral": 20, "negative": 10}
            community_response = "supportive"
        elif sentiment_score >= 0.6:
            sentiment_distribution = {"positive": 60, "neutral": 30, "negative": 10}
            community_response = "supportive"
        elif sentiment_score >= 0.4:
            sentiment_distribution = {"positive": 30, "neutral": 50, "negative": 20}
            community_response = "mixed"
        elif sentiment_score >= 0.3:
            sentiment_distribution = {"positive": 20, "neutral": 30, "negative": 50}
            community_response = "critical"
        else:
            sentiment_distribution = {"positive": 10, "neutral": 20, "negative": 70}
            community_response = "critical"

        return {
            "sentiment_score": sentiment_score,
            "engagement_score": engagement_score,
            "key_topics": key_topics,
            "positive_aspects": positive_aspects,
            "negative_aspects": negative_aspects,
            "key_emotions": key_emotions,
            "sentiment_distribution": sentiment_distribution,
            "community_response": community_response,
        }

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not context.get("text", ""):
                print("No text provided for biodiversity analysis")
                return self._get_default_results()

            input_text = context["text"]

            # Extract metadata from the text if available
            author_types = []
            locations = []
            dates = []

            # Extract author info, locations, and dates if present in structured format
            author_pattern = re.compile(r"Author(?:[\s:]*)([^\n]+)")
            location_pattern = re.compile(r"Location(?:[\s:]*)([^\n]+)")
            date_pattern = re.compile(r"Date(?:[\s:]*)([^\n]+)")

            author_matches = author_pattern.findall(input_text)
            location_matches = location_pattern.findall(input_text)
            date_matches = date_pattern.findall(input_text)

            if author_matches:
                author_types = list(set(author_matches))
            if location_matches:
                locations = list(set(location_matches))
            if date_matches:
                dates = sorted(list(set(date_matches)))

            # Create time span if dates are available
            time_span = ""
            if len(dates) >= 2:
                time_span = f"From {dates[0]} to {dates[-1]}"
            elif len(dates) == 1:
                time_span = dates[0]

            prompt = self._create_prompt_template()
            chain = prompt | self.llm

            response = chain.invoke(
                {"text": input_text}, config={"callbacks": [self.callback_handler]}
            ).content.strip()

            # Add validation before processing
            if not self._validate_response(response):
                print(f"Invalid response format from biodiversity agent: {response}")
                default_results = self._get_default_results()
                default_results.update({
                    "author_types": author_types,
                    "locations": locations,
                    "time_span": time_span,
                })
                return default_results

            results = self._process_response(response)

            # Add the additional metadata
            results.update({
                "author_types": author_types,
                "locations": locations,
                "time_span": time_span,
            })

            return results

        except Exception as e:
            print(f"Error in biodiversity analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._get_default_results()


class BiodiversityFrameworkAgent(AnalysisAgent):
    """Specialized agent for analyzing text against the biodiversity framework"""

    def __init__(self, llm):
        super().__init__(llm)
        self.expected_parts = 3  # N, S, M values

    def _create_prompt_template(self) -> PromptTemplate:
        template = """
        Analyze the following text in relation to biodiversity framework question:
        
        QUESTION: {question}
        
        TEXT:
        {text}
        
        Based on a comprehensive analysis of the text, evaluate:
        
        1. How many SPECIFIC and DISTINCT sentences/paragraphs directly address the question? Be precise and count EXACT occurrences.
           * A relevant mention must EXPLICITLY address the specific question asked
           * Count each distinct relevant sentence/paragraph EXACTLY once
           * DO NOT inflate this number - it must accurately reflect the text
           * If no relevant mentions exist, N should be 0
           * Scrutinize each potential mention carefully and count ONLY direct responses to the question
           * Relevant mentions should contain substantive information, not just keyword matches
        
        2. How specific and detailed (S) are these mentions on a scale of 0-100?
           * Consider detail level, actionable information, concrete facts
           * Higher scores for specific data, metrics, or detailed plans
           * Lower scores for vague statements or general principles
        
        3. What percentage of mentions are redundant or repetitive (M) on a scale of 0-100?
           * Identify repeated information across mentions
           * Higher scores indicate more redundancy and repetition
        
        YOU MUST RESPOND IN EXACTLY THIS FORMAT WITH 3 PARTS SEPARATED BY |:
        <N>|<S>|<M>
        
        WHERE:
        - <N>: EXACT count of relevant sentences/mentions directly addressing the question (integer)
        - <S>: Specificity rating on scale 0-100 (how detailed and actionable)
        - <M>: Multiplicity/redundancy percentage on scale 0-100 (how repetitive)
        
        IMPORTANT:
        - Be extremely careful with the N count - it should be accurate and verifiable
        - Only count mentions that substantively address the question
        - Avoid defaulting to arbitrary values (like 5) - count the actual mentions
        - Your N value should be a number that someone can manually verify in the text
        - For longer texts, analyze each sentence individually to avoid overcounting
        - The system is parsing the document into individual sentences to verify your count
        
        EXAMPLE RESPONSES:
        0|0|0 (If no mentions found)
        3|65|10 (3 relevant mentions with good specificity and low redundancy)
        12|80|40 (12 mentions with high specificity but significant redundancy)
        
        RESPOND WITH ONLY THE FORMATTED TEXT. NO OTHER TEXT OR EXPLANATION.
        """
        return PromptTemplate(template=template, input_variables=["question", "text"])

    def _get_default_results(self) -> Dict[str, Any]:
        return {
            "N": 0,
            "S": 0.0,
            "M": 0.0,
        }

    def _process_response(self, response: str) -> Dict[str, Any]:
        parts = self._safe_split(response)

        try:
            n_value = int(parts[0])
            if n_value < 0:
                n_value = 0
        except (ValueError, IndexError):
            n_value = 0

        try:
            s_value = float(parts[1])
            if not 0 <= s_value <= 100:
                s_value = 0.0
        except (ValueError, IndexError):
            s_value = 0.0

        try:
            m_value = float(parts[2])
            if not 0 <= m_value <= 100:
                m_value = 0.0
        except (ValueError, IndexError):
            m_value = 0.0

        return {
            "N": n_value,
            "S": s_value,
            "M": m_value,
        }

    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print("BiodiversityFrameworkAgent: Starting analysis...")
            prompt = self._create_prompt_template()
            chain = prompt | self.llm

            # Ensure we have both text and question
            if "text" not in context or "question" not in context:
                print(
                    "BiodiversityFrameworkAgent: Missing required context - text or question missing"
                )
                return self._get_default_results()

            # Get the text and question
            text = context["text"]
            question = context["question"]

            # Truncate text if it's too long (over 12,000 characters)
            max_text_length = 12000
            original_length = len(text)
            if original_length > max_text_length:
                print(
                    f"BiodiversityFrameworkAgent: Truncating text from {original_length} to {max_text_length} characters"
                )
                # Take the first third and last two thirds of the allowed length to get relevant parts
                first_part = text[: int(max_text_length * 0.33)]
                last_part = text[-(int(max_text_length * 0.67)) :]
                text = first_part + "\n...[text truncated]...\n" + last_part

            print(
                f"BiodiversityFrameworkAgent: Analyzing question: {question[:100]}..."
            )
            print(f"BiodiversityFrameworkAgent: Text length: {len(text)} characters")

            response = chain.invoke(
                {"text": text, "question": question},
                config={"callbacks": [self.callback_handler]},
            ).content.strip()

            print(f"BiodiversityFrameworkAgent: Got response: {response}")

            # Validate response format
            if not self._validate_response(response):
                print(
                    f"BiodiversityFrameworkAgent: Invalid response format: {response}"
                )
                return self._get_default_results()

            results = self._process_response(response)
            print(
                f"BiodiversityFrameworkAgent: Processed results: N={results['N']}, S={results['S']}, M={results['M']}"
            )
            return results

        except Exception as e:
            print(f"BiodiversityFrameworkAgent: Error in analysis: {str(e)}")
            import traceback

            traceback.print_exc()
            return self._get_default_results()


def search_project_forums_and_reviews(project_folder_name: str, max_results: int = 10) -> str:
    """
    Search forums and Google reviews for biodiversity projects using GoogleSerperAPIWrapper
    
    Args:
        project_folder_name: Name of the project folder
        max_results: Maximum number of results to fetch
        
    Returns:
        Formatted string with forum and review data
    """
  
    
    # Check for API key
    serper_api_key = os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        return "SERPER API key not found in environment variables"
    
    # Extract clean project name
    project_name = project_folder_name.split("_")[0] if "_" in project_folder_name else project_folder_name
    print(f"Searching forums and reviews for project: {project_name}")
    print(f"Using API key: {serper_api_key[:10]}...")
    
    # Test the API key first
    print("Testing API key validity...")
    try:
        import requests
        test_response = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": serper_api_key,
                "Content-Type": "application/json"
            },
            json={"q": "test"},
            timeout=10
        )
        if test_response.status_code == 403:
            return f"API Key is invalid or expired. Status: {test_response.status_code}. Please get a new key from https://serper.dev/"
        elif test_response.status_code != 200:
            print(f"API test returned status {test_response.status_code}: {test_response.text}")
        else:
            print("API key is valid!")
    except Exception as e:
        print(f"API key test failed: {e}")
    
    # Initialize GoogleSerperAPIWrapper with Australia focus and more results
    try:
        search = GoogleSerperAPIWrapper(
            serper_api_key=serper_api_key,
            gl="au",  # Australia geolocation
            hl="en",  # English language
            k=10      # Get more results per query (default is usually 10, but being explicit)
        )
    except Exception as e:
        print(f"Failed to initialize GoogleSerperAPIWrapper: {e}")
        return f"Failed to initialize search tool: {str(e)}"
    
    # Comprehensive Australia-focused search queries for more content
    search_queries = [
        f'"{project_name}" Australia news environmental impact',
        f'"{project_name}" community forum discussion environmental concerns',
        f'"{project_name}" social media reaction community response',
        f'"{project_name}" environmental assessment report Australia',
        f'"{project_name}" local community consultation feedback'
    ]
    
    all_content = []
    
    for query in search_queries:
        try:
            print(f"  Searching: {query}")
            
            # Perform search
            results = search.run(query)
            print(f"    Search completed, processing results...")
            
            if results and len(results) > 20:  # Only if we got some content
                print(f"    Raw results length: {len(results)} characters")
                
                # Simple text processing - just capture whatever we get
                # Determine source type from query
                if "news" in query.lower():
                    source_type = "news"
                elif "forum" in query.lower():
                    source_type = "community_forums" 
                elif "social media" in query.lower():
                    source_type = "social_media"
                else:
                    source_type = "community_forums"
                
                # Create a content block with ALL the results for this query (no truncation)
                content_block = f"""Source: {source_type}
Title: Search Results for "{project_name}" - {source_type.replace('_', ' ').title()}
Content: {results}
URL: Multiple sources from Google search
Query_Used: {query}
Date_Searched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location_Focus: Australia
Relevance: High (contains '{project_name}')
---

"""
                all_content.append(content_block)
                print(f"    Added {source_type} results block ({len(results)} chars)")
            else:
                print(f"    No substantial results for query: {query}")
            
        except Exception as e:
            print(f"  Error searching '{query}': {str(e)}")
            continue
    
    # Check if we got real results
    print(f"Total content blocks collected: {len(all_content)}")
    
    if not all_content:
        print("No real results found, providing mock data for demonstration")
        mock_content = f"""Source: Web Search
Title: {project_name} - Community Discussions
Content: This project has been discussed in various environmental forums. Community members have shared concerns about biodiversity impacts and conservation measures. The project appears to have mixed reception from environmental groups with some supporting renewable energy initiatives while others express concerns about habitat disruption.
Query: Sample search
Relevance: Moderate (general environmental discussion)
---

Source: Google Reviews
Title: {project_name} - Environmental Impact Reviews
Content: Online reviews and discussions about this project's environmental impact. Users have commented on both positive aspects (renewable energy) and concerns (wildlife habitat effects). The project has generated significant community interest and debate.
Query: Sample search
Relevance: High (direct project references)
---"""
        
        return f"Project: {project_name}\n\n{mock_content}\n\nNote: This is sample data for demonstration. Real search failed."
    
    print(f"Returning {len(all_content)} real search results")
    return f"Project: {project_name}\n\n" + "\n\n".join(all_content[:max_results])
