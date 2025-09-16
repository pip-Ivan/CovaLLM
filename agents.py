from abc import ABC, abstractmethod
from typing import Dict, Any, List
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.schema import LLMResult
import re
import os
import logging

from datetime import datetime
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import io, contextlib


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


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
        """Calculate cost with better model name matching and fallback handling."""
        if not isinstance(usage, dict):
            return 0.0
            
        # Updated cost per 1K tokens with more comprehensive model coverage
        rates = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},  # GPT-4 Turbo alternative name
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # Updated rates
            "deepseek-chat": {"input": 0.00014, "output": 0.00028},  # Updated rates
            "deepseek-coder": {"input": 0.00014, "output": 0.00028},
        }
        
        model_name = usage.get("model", "gpt-4o-mini")
        
        # Normalize model name - handle variations
        model_key = model_name.lower().strip()
        
        # Try exact match first
        rate = rates.get(model_key)
        
        # If no exact match, try partial matching for known patterns
        if rate is None:
            for known_model in rates.keys():
                if known_model in model_key or model_key in known_model:
                    rate = rates[known_model]
                    break
        
        # Fallback to gpt-4o-mini rates if model not found
        if rate is None:
            rate = rates["gpt-4o-mini"]
            logging.warning(f"Unknown model '{model_name}', using gpt-4o-mini rates for cost calculation")
        
        # Validate token counts
        prompt_tokens = max(0, usage.get("prompt_tokens", 0))
        completion_tokens = max(0, usage.get("completion_tokens", 0))
        
        if not isinstance(prompt_tokens, (int, float)) or not isinstance(completion_tokens, (int, float)):
            return 0.0
            
        try:
            cost = (prompt_tokens * rate["input"] + completion_tokens * rate["output"]) / 1000
            return max(0.0, cost)  # Ensure non-negative cost
        except (KeyError, TypeError, ZeroDivisionError):
            return 0.0


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

    

    def create_prompt_template(self) -> PromptTemplate:
     template = """
        You are analyzing a report against a Kunming–Montreal Global Biodiversity Framework (GBF) question.

        QUESTION: {question}

        TEXT TO ANALYZE:
        {text}

        DEFINITIONS
        - Unit: One sentence OR one bullet/line item.
        - Mention: A Unit that directly answers or relates to the QUESTION (paraphrases allowed).
        - Evidence Fields (binary): For each unique Mention, evaluate 5 fields as present (1) or absent (0).

        ANALYSIS TASKS

        STEP 1 — IDENTIFY MENTIONS (N)
        Count all Units that directly answer or relate to the QUESTION.
        - Include sentences that address the question topic.
        - Include bullet points or line items that are relevant.
        - Count each Unit separately, even if similar.
        - If no relevant Units are found, set N = 0.

        STEP 2 — DEDUPLICATE BY FACTUAL CONTENT (M)
        Merge Mentions that are factually equivalent:
        - Combine paraphrases stating the same facts.
        - Merge mentions with the same actions, targets, or key values (numbers/timeframes).
        - Keep one representative Unit per unique fact cluster.
        - M = number of unique factual Mentions after deduplication.

        STEP 3 — EVIDENCE SCORING (STRICTLY BINARY)
        For each unique Mention, output five binary values (0 or 1), one per field:

        1) action_commitment: Is there a concrete action or commitment? (0/1)
        2) numeric_metric_threshold: Is a measurable number/percentage/threshold given? (0/1)
        3) framework_tag: Is a recognized framework/standard explicitly named? (0/1)
        4) timeframe: Is an explicit date/year/period/frequency given? (0/1)
        5) method_baseline_evidence: Is a method, baseline, or evidence approach mentioned? (0/1)

        STRICTNESS RULES
        - When uncertain about duplication, merge into one unique item.
        - If a field is questionable or unclear, score it 0.
        - Do not infer or assume unstated details; rely only on explicit text.

        RESPONSE FORMAT (single line, no explanation)
        N|M|binary_vectors 

        Where:
        - N = total relevant Units (integer).
        - M = unique factual Mentions (integer).
        - binary_vectors  = comma-separated list of 5-digit binary vectors for each Mention.
        Example: "10110,01000,11100" (one 5-digit vector per Mention, in order).

        EDGE CASES
        - If N = 0, return "0|0|".
        - If M = 0 after deduplication, return "N|0|".

        EXAMPLES
        5|3|10110,01000,11100
        0|0|
        8|6|10000,10010,11000,11100,01000,00100
        1|1|11110
        """
     return PromptTemplate(template=template, input_variables=["question", "text"])


    def _get_default_results(self) -> Dict[str, Any]:
        return {
            "N": 0,      # Total Units count
            "M": 0,      # Unique Mentions count  
            "S": 0,    # Evidence score percentage
        }

    def _process_response(self, response: str) -> Dict[str, Any]:
        parts = self._safe_split(response)

        try:
            n_value = int(parts[0]) if int(parts[0]) >= 0 else 0
        except (ValueError, IndexError):
            n_value = 0

        try:
            m_value = int(parts[1]) if int(parts[1]) >= 0 else 0
        except (ValueError, IndexError):
            m_value = 0

        binary_vectors = parts[2] if len(parts) > 2 else ""
        vectors = [v.strip() for v in binary_vectors.split(",") if v.strip()]

        # S = sum of all 1s across vectors
        S = sum(sum(int(ch) for ch in v if ch in "01") for v in vectors)

        return {"N": n_value, "M": m_value, "S": S}


    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze text against biodiversity framework with retry logic and validation."""
        try:
            logging.debug("BiodiversityFrameworkAgent: Starting analysis...")
            
            # Input validation
            if not isinstance(context, dict):
                logging.error("BiodiversityFrameworkAgent: Context must be a dictionary")
                return self._get_default_results()
                
            # Ensure we have both text and question
            if "text" not in context or "question" not in context:
                logging.error("BiodiversityFrameworkAgent: Missing required context - text or question missing")
                return self._get_default_results()

            text = context.get("text", "").strip()
            question = context.get("question", "").strip()
            
            # Validate inputs
            if not text or not question:
                logging.warning("BiodiversityFrameworkAgent: Empty text or question provided")
                return self._get_default_results()
                
            if not isinstance(text, str) or not isinstance(question, str):
                logging.error("BiodiversityFrameworkAgent: Text and question must be strings")
                return self._get_default_results()

            # Truncate text if it's too long (over 12,000 characters)
            max_text_length = 12000
            original_length = len(text)
            if original_length > max_text_length:
                logging.info(
                    f"BiodiversityFrameworkAgent: Truncating text from {original_length} to {max_text_length} characters"
                )
                # Take the first third and last two thirds of the allowed length to get relevant parts
                first_part = text[:int(max_text_length * 0.33)]
                last_part = text[-(int(max_text_length * 0.67)):]
                text = first_part + "\n...[text truncated]...\n" + last_part

            logging.debug(f"BiodiversityFrameworkAgent: Analyzing question: {question[:100]}...")
            logging.debug(f"BiodiversityFrameworkAgent: Text length: {len(text)} characters")

            # Create chain with error handling
            prompt = self.create_prompt_template()
            if not prompt:
                logging.error("BiodiversityFrameworkAgent: Failed to create prompt template")
                return self._get_default_results()
                
            chain = prompt | self.llm

            # Invoke with timeout and callbacks
            response = chain.invoke(
                {"text": text, "question": question},
                config={"callbacks": [self.callback_handler]},
            )
            
            if not response or not hasattr(response, 'content'):
                logging.error("BiodiversityFrameworkAgent: Invalid response from LLM")
                return self._get_default_results()
                
            response_content = response.content.strip()
            logging.debug(f"BiodiversityFrameworkAgent: Got response: {response_content}")

            # Validate response format
            if not self._validate_response(response_content):
                logging.warning(f"BiodiversityFrameworkAgent: Invalid response format: {response_content}")
                return self._get_default_results()

            results = self._process_response(response_content)
            
            # Validate processed results
            if not self._validate_results(results):
                logging.warning("BiodiversityFrameworkAgent: Invalid processed results")
                return self._get_default_results()
                
            logging.debug(
                f"BiodiversityFrameworkAgent: Processed results: N={results['N']}, S={results['S']}, M={results['M']}"
            )
            return results

        except Exception as e:
            logging.error(f"BiodiversityFrameworkAgent: Error in analysis: {str(e)}")
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                import traceback
                traceback.print_exc()
            return self._get_default_results()
            
    def _validate_results(self, results: Dict[str, Any]) -> bool:
        """Validate that processed results match exact definitions."""
        if not isinstance(results, dict):
            return False
            
        required_keys = ["N", "S", "M"]
        for key in required_keys:
            if key not in results:
                return False
                
        # Validate N (should be non-negative integer)
        n_val = results.get("N", -1)
        if not isinstance(n_val, int) or n_val < 0:
            return False
            
        # Validate S (should be 1-100 percentage)
        s_val = results.get("S", 0)
        if not isinstance(s_val, (int, float)) or s_val < 1.0 or s_val > 100.0:
            return False
            
        # Validate M (should be 0-100 percentage)
        m_val = results.get("M", -1)
        if not isinstance(m_val, (int, float)) or m_val < 0.0 or m_val > 100.0:
            return False
                
        return True


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_project_forums_and_reviews(project_folder_name: str, max_results: int = 10) -> str:
    """
    Search forums and Google reviews for biodiversity projects using GoogleSerperAPIWrapper
    
    Args:
        project_folder_name: Name of the project folder
        max_results: Maximum number of results to fetch
        
    Returns:
        Formatted string with forum and review data
    """
    
    # Input validation
    if not project_folder_name or not isinstance(project_folder_name, str):
        logging.error("Invalid project folder name provided")
        return "Error: Invalid project folder name"
        
    if not isinstance(max_results, int) or max_results <= 0:
        max_results = 10
        
    # Check for API key
    serper_api_key = os.getenv('SERPER_API_KEY')
    if not serper_api_key:
        logging.warning("SERPER API key not found in environment variables")
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
           # print(f"  Searching: {query}")
            
            # Perform search
            with contextlib.redirect_stdout(io.StringIO()):
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
            #    print(f"    Added {source_type} results block ({len(results)} chars)")
            else:
                print(f"    No substantial results for query: {query}")
            
        except Exception as e:
          #  print(f"  Error searching '{query}': {str(e)}")
            continue
    
    # Check if we got real results
   # print(f"Total content blocks collected: {len(all_content)}")
    
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
    
   # print(f"Returning {len(all_content)} real search results")
    return f"Project: {project_name}\n\n" + "\n\n".join(all_content[:max_results])

