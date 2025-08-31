from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Set, Union
from enum import Enum
import csv
from io import StringIO
import re
import nltk

nltk.download('punkt_tab')
nltk.download('punkt')  
nltk.download('stopwords')


class BiodiversityCategory(str, Enum):
    PLANNING = "Planning and Management"
    CONSERVATION = "Conservation and Restoration"
    DECISION = "Decision-Making and Disclosure"
    TECHNOLOGY = "Technology, Knowledge, and Collaboration"
    FINANCIAL = "Financial Support and Investment"


class Question(BaseModel):
    text: str
    relevant_excerpts: List[str] = []
    # Add question-level scoring metrics
    N: int = 0  # Number of relevant mentions
    S: float = 0.0  # Specificity score (0-100)
    M: float = 0.0  # Multiplicity/repetition score (0-100)

    def calculate_score(self) -> float:
        """Calculate question score using N × S × (1 − M) formula"""
        # Normalize values to 0-1 range
        norm_n = min(self.N / 10, 1.0)  # Cap at 10 mentions
        norm_s = self.S / 100
        norm_m = 1 - (self.M / 100)  # Invert M score (lower is better)

        # Direct formula from requirements
        return norm_n * norm_s * norm_m

    def add_relevant_excerpt(self, excerpt: str) -> None:
        """Add a relevant excerpt and update N count"""
        self.relevant_excerpts.append(excerpt)
        self.N += 1

    def update_specificity(self, value: float) -> None:
        """Update specificity score (0-100)"""
        self.S = max(0, min(100, value))

    def update_multiplicity(self, value: float) -> None:
        """Update multiplicity score (0-100)"""
        self.M = max(0, min(100, value))


class Subcategory(BaseModel):
    id: str
    name: str
    questions: List[Question]
    category: BiodiversityCategory

    # Scoring metrics at subcategory level (will be calculated from questions)
    N: int = 0
    S: float = 0.0
    M: float = 0.0
    
    # Dollar value allocation for this subcategory
    dollar_value: float = 0.0

    def calculate_score(self) -> float:
        """Calculate subcategory score as average of question scores"""
        if not self.questions:
            return 0.0
        return sum(q.calculate_score() for q in self.questions) / len(self.questions)

    def add_relevant_excerpt(self, question_index: int, excerpt: str) -> None:
        """Add a relevant excerpt to a specific question"""
        if 0 <= question_index < len(self.questions):
            self.questions[question_index].add_relevant_excerpt(excerpt)
            # Update subcategory N as sum of question Ns
            self.update_from_questions()

    def update_from_questions(self) -> None:
        """Update subcategory metrics based on question metrics"""
        if not self.questions:
            return

        # N is sum of all question Ns
        self.N = sum(q.N for q in self.questions)

        # S and M are weighted averages based on question N values
        total_n = self.N
        if total_n > 0:
            self.S = sum(q.S * q.N for q in self.questions) / total_n
            self.M = sum(q.M * q.N for q in self.questions) / total_n
        else:
            self.S = 0.0
            self.M = 0.0

    def update_specificity(self, value: float) -> None:
        """Legacy method - now updates all questions with same value"""
        for question in self.questions:
            question.update_specificity(value)
        self.S = value  # Still update subcategory value for backward compatibility

    def update_multiplicity(self, value: float) -> None:
        """Legacy method - now updates all questions with same value"""
        for question in self.questions:
            question.update_multiplicity(value)
        self.M = value  # Still update subcategory value for backward compatibility


class CategoryScore(BaseModel):
    category: BiodiversityCategory
    subcategories: List[Subcategory]
    weight: float = 1.0  # Default weight for category

    # Category-level scoring metrics (will be calculated from subcategories)
    N: int = 0
    S: float = 0.0
    M: float = 0.0

    def get_total_score(self) -> float:
        """Calculate average score across all subcategories"""
        if not self.subcategories:
            return 0.0
        return sum(sc.calculate_score() for sc in self.subcategories) / len(
            self.subcategories
        )

    def calculate_score(self) -> float:
        """Calculate category score as average of subcategory scores (same as get_total_score)"""
        return self.get_total_score()

    def update_from_subcategories(self) -> None:
        """Update category metrics based on subcategory values"""
        if not self.subcategories:
            return

        # Sum N values across all subcategories
        self.N = sum(subcategory.N for subcategory in self.subcategories)

        # Average S and M values across all subcategories (weighted by N)
        if self.N > 0:
            self.S = (
                sum(subcategory.S * subcategory.N for subcategory in self.subcategories)
                / self.N
            )
            self.M = (
                sum(subcategory.M * subcategory.N for subcategory in self.subcategories)
                / self.N
            )
        else:
            self.S = 0.0
            self.M = 0.0

    def get_total_dollar_value(self) -> float:
        """Calculate total dollar value for this category"""
        return sum(subcategory.dollar_value for subcategory in self.subcategories)


class BiodiversityReport(BaseModel):
    category_scores: List[CategoryScore]
    additional_insights: Dict[str, Any] = {
        "sentiment": {
            "overall_score": 0.0,
            "positive_aspects": set(),
            "negative_aspects": set(),
            "key_emotions": set(),
        },
        "stakeholder": {
            "total_stakeholders": set(),
            "engagement_level": 0.0,
            "main_concerns": set(),
        },
        "biodiversity": {
            "species_count": 0,
            "habitat_types": set(),
            "conservation_status": {},
            "impact_score": 0.0,
        },
        "social_media": {
            "engagement_score": 0.0,
            "sentiment_distribution": {},
            "key_topics": set(),
        },
    }

    def get_overall_score(
        self, weights: Optional[Dict[BiodiversityCategory, float]] = None
    ) -> float:
        """
        Calculate overall weighted score across all categories.
        
        The scoring system works as follows:
        1. Question scores use: N × S × (1 - M) formula
        2. Subcategory scores are averages of their question scores
        3. Category scores are averages of their subcategory scores
        4. Overall score is a weighted average of category scores using normalized weights
        
        Args:
            weights: Dictionary mapping BiodiversityCategory to weight values.
                    If None, equal weights (1.0) are used for all categories.
                    Weights are automatically normalized to sum to 1.0.
        
        Returns:
            Float between 0 and 1 representing the overall biodiversity score
        """
        if not self.category_scores:
            return 0.0

        if weights is None:
            weights = {category: 1.0 for category in BiodiversityCategory}

        # Normalize weights to sum to 1
        normalized_weights = self._normalize_weights(weights)

        weighted_score = 0.0
        for cs in self.category_scores:
            category_weight = normalized_weights.get(cs.category, 0.0)
            category_score = cs.calculate_score()
            weighted_score += category_score * category_weight

        return weighted_score

    def _normalize_weights(self, weights: Dict[BiodiversityCategory, float]) -> Dict[BiodiversityCategory, float]:
        """Normalize weights so they sum to 1"""
        # Get weights for categories that actually exist in the report
        existing_weights = {}
        for cs in self.category_scores:
            existing_weights[cs.category] = weights.get(cs.category, 1.0)
        
        # Calculate sum and normalize
        weight_sum = sum(existing_weights.values())
        if weight_sum == 0:
            # If all weights are 0, assign equal weights
            equal_weight = 1.0 / len(existing_weights) if existing_weights else 0.0
            return {cat: equal_weight for cat in existing_weights.keys()}
        
        # Normalize so they sum to 1
        return {cat: weight / weight_sum for cat, weight in existing_weights.items()}

    def get_category_weights_info(self, weights: Optional[Dict[BiodiversityCategory, float]] = None) -> Dict[str, Any]:
        """Get information about category weights for display purposes"""
        if weights is None:
            weights = {category: 1.0 for category in BiodiversityCategory}
        
        normalized_weights = self._normalize_weights(weights)
        
        # Return info about weights
        weight_info = {}
        for cs in self.category_scores:
            weight_info[cs.category.value] = {
                "original_weight": weights.get(cs.category, 1.0),
                "normalized_weight": normalized_weights.get(cs.category, 0.0),
                "score": cs.calculate_score(),
                "weighted_contribution": cs.calculate_score() * normalized_weights.get(cs.category, 0.0)
            }
        
        return weight_info

    def get_total_dollar_value(self) -> float:
        """Calculate total dollar value across all categories"""
        return sum(cs.get_total_dollar_value() for cs in self.category_scores)

    def get_dollar_value_info(self) -> Dict[str, Any]:
        """Get comprehensive dollar value information for display purposes"""
        dollar_info = {
            "categories": {},
            "subcategories": {},
            "total": self.get_total_dollar_value(),
            "category_order": [],
            "subcategory_order": []
        }
        
        # Maintain the order from the framework definition
        for category_score in self.category_scores:
            category_name = category_score.category.value
            category_total = category_score.get_total_dollar_value()
            
            dollar_info["categories"][category_name] = {
                "total": category_total,
                "subcategories": []
            }
            dollar_info["category_order"].append(category_name)
            
            for subcategory in category_score.subcategories:
                subcategory_info = {
                    "id": subcategory.id,
                    "name": subcategory.name,
                    "value": subcategory.dollar_value,
                    "category": category_name
                }
                dollar_info["subcategories"][subcategory.id] = subcategory_info
                dollar_info["subcategory_order"].append(subcategory.id)
                dollar_info["categories"][category_name]["subcategories"].append(subcategory_info)
        
        return dollar_info

    def update_insights(self, subcategory_id: str, insights: Dict[str, Any]) -> None:
        """Update additional insights from specialized agents"""
        count = 0

        # Aggregate sentiment insights
        if "sentiment_score" in insights:
            count += 1
            self.additional_insights["sentiment"]["overall_score"] += insights[
                "sentiment_score"
            ]
            self.additional_insights["sentiment"]["positive_aspects"].update(
                insights.get("positive_aspects", [])
            )
            self.additional_insights["sentiment"]["negative_aspects"].update(
                insights.get("negative_aspects", [])
            )
            self.additional_insights["sentiment"]["key_emotions"].update(
                insights.get("key_emotions", [])
            )

        # Aggregate stakeholder insights
        if "stakeholder_count" in insights:
            count += 1
            self.additional_insights["stakeholder"]["total_stakeholders"].update(
                insights.get("key_stakeholders", [])
            )
            self.additional_insights["stakeholder"]["engagement_level"] += insights[
                "engagement_level"
            ]
            self.additional_insights["stakeholder"]["main_concerns"].update(
                insights.get("main_concerns", [])
            )

        # Aggregate biodiversity insights
        if "species_count" in insights:
            count += 1
            self.additional_insights["biodiversity"]["species_count"] = max(
                self.additional_insights["biodiversity"]["species_count"],
                insights["species_count"],
            )
            self.additional_insights["biodiversity"]["habitat_types"].update(
                insights.get("habitat_types", [])
            )

            # Merge conservation status dictionaries
            for status, status_count in insights.get("conservation_status", {}).items():
                self.additional_insights["biodiversity"]["conservation_status"][
                    status
                ] = (
                    self.additional_insights["biodiversity"]["conservation_status"].get(
                        status, 0
                    )
                    + status_count
                )

            self.additional_insights["biodiversity"]["impact_score"] += insights.get(
                "impact_score", 0
            )

        # Aggregate social media insights
        if "engagement_score" in insights:
            count += 1
            self.additional_insights["social_media"]["engagement_score"] += insights[
                "engagement_score"
            ]

            # Merge sentiment distributions
            for sentiment, value in insights.get("sentiment_distribution", {}).items():
                self.additional_insights["social_media"]["sentiment_distribution"][
                    sentiment
                ] = (
                    self.additional_insights["social_media"][
                        "sentiment_distribution"
                    ].get(sentiment, 0)
                    + value
                )

            self.additional_insights["social_media"]["key_topics"].update(
                insights.get("key_topics", [])
            )

        # Note: We'll normalize these values at the end when generating the final report

    def normalize_insights(self) -> None:
        """Normalize aggregate insights for final reporting"""
        # Calculate number of subcategories with insights
        insight_count = sum(1 for cs in self.category_scores for _ in cs.subcategories)

        if insight_count > 0:
            # Normalize sentiment scores
            self.additional_insights["sentiment"]["overall_score"] /= insight_count

            # Normalize stakeholder engagement level
            self.additional_insights["stakeholder"]["engagement_level"] /= insight_count

            # Normalize biodiversity impact score
            self.additional_insights["biodiversity"]["impact_score"] /= insight_count

            # Normalize social media engagement score
            self.additional_insights["social_media"][
                "engagement_score"
            ] /= insight_count

            # Normalize sentiment distribution
            total = sum(
                self.additional_insights["social_media"][
                    "sentiment_distribution"
                ].values()
            )
            if total > 0:
                self.additional_insights["social_media"]["sentiment_distribution"] = {
                    k: v / total * 100
                    for k, v in self.additional_insights["social_media"][
                        "sentiment_distribution"
                    ].items()
                }

        # Convert sets to lists for JSON serialization
        for category in self.additional_insights.values():
            for key, value in category.items():
                if isinstance(value, set):
                    category[key] = list(value)

    def generate_csv_report(self, weights: Optional[Dict[BiodiversityCategory, float]] = None) -> str:
        """Generate a CSV report of the biodiversity assessment scores"""
        output = StringIO()
        writer = csv.writer(output)

        # Write header with expanded columns for question-level metrics, weights, and dollar values
        writer.writerow(
            [
                "Category",
                "Category Weight (Original)",
                "Category Weight (Normalized)",
                "Category Dollar Total",
                "Category N",
                "Category S (%)",
                "Category M (%)",
                "Category Score",
                "Weighted Contribution",
                "Subcategory",
                "Subcategory Dollar Value",
                "Subcategory N",
                "Subcategory S (%)",
                "Subcategory M (%)",
                "Subcategory Score",
                "Question",
                "Question N",
                "Question S (%)",
                "Question M (%)",
                "Question Score",
            ]
        )

        # Calculate normalized weights
        if weights is None:
            weights = {category: 1.0 for category in BiodiversityCategory}
        normalized_weights = self._normalize_weights(weights)

        # Write data for each category, subcategory and question
        for category_score in self.category_scores:
            # Update category metrics from subcategories
            category_score.update_from_subcategories()
            category_name = category_score.category.value
            
            # Get weight and dollar information for this category
            original_weight = weights.get(category_score.category, 1.0)
            normalized_weight = normalized_weights.get(category_score.category, 0.0)
            weighted_contribution = category_score.calculate_score() * normalized_weight
            category_dollar_total = category_score.get_total_dollar_value()
            
            category_row_written = False

            for subcategory in category_score.subcategories:
                subcategory_row_written = False

                for i, question in enumerate(subcategory.questions):
                    # For the first question of the first subcategory, include category and subcategory data
                    if not category_row_written and not subcategory_row_written:
                        writer.writerow(
                            [
                                category_name,
                                f"{original_weight:.2f}",
                                f"{normalized_weight:.4f}",
                                f"{category_dollar_total:.0f}",
                                category_score.N,
                                f"{category_score.S:.1f}",
                                f"{category_score.M:.1f}",
                                f"{category_score.calculate_score():.4f}",
                                f"{weighted_contribution:.4f}",
                                subcategory.name,
                                f"{subcategory.dollar_value:.0f}",
                                subcategory.N,
                                f"{subcategory.S:.1f}",
                                f"{subcategory.M:.1f}",
                                f"{subcategory.calculate_score():.4f}",
                                # Question data
                                f"Q{i+1}: {question.text}",
                                question.N,
                                f"{question.S:.1f}",
                                f"{question.M:.1f}",
                                f"{question.calculate_score():.4f}",
                            ]
                        )
                        category_row_written = True
                        subcategory_row_written = True
                    # For the first question of subsequent subcategories, include subcategory data but not category data
                    elif not subcategory_row_written:
                        writer.writerow(
                            [
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",  # Empty category columns (including weights and dollar total)
                                subcategory.name,
                                f"{subcategory.dollar_value:.0f}",
                                subcategory.N,
                                f"{subcategory.S:.1f}",
                                f"{subcategory.M:.1f}",
                                f"{subcategory.calculate_score():.4f}",
                                # Question data
                                f"Q{i+1}: {question.text}",
                                question.N,
                                f"{question.S:.1f}",
                                f"{question.M:.1f}",
                                f"{question.calculate_score():.4f}",
                            ]
                        )
                        subcategory_row_written = True
                    else:
                        # For subsequent questions, only include question data
                        writer.writerow(
                            [
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",  # Empty category columns (including weights and dollar total)
                                "",
                                "",
                                "",
                                "",
                                "",
                                "",  # Empty subcategory columns (including dollar value)
                                f"Q{i+1}: {question.text}",
                                question.N,
                                f"{question.S:.1f}",
                                f"{question.M:.1f}",
                                f"{question.calculate_score():.4f}",
                            ]
                        )

        return output.getvalue()


# Define the evaluation framework with all questions
BIODIVERSITY_FRAMEWORK = [
    # 1. Planning and Management
    CategoryScore(
        category=BiodiversityCategory.PLANNING,
        subcategories=[
            Subcategory(
                id="ecosystem_species_management",
                name="Ecosystem/Species Management Plan",
                category=BiodiversityCategory.PLANNING,
                questions=[
                    Question(
                        text="Are there clear actions to reduce negative impacts on biodiversity?"
                    ),
                    Question(
                        text="Are there goals and progress updates for ecosystem restoration and rehabilitation, including stakeholder engagement?"
                    ),
                    Question(
                        text="Are measures to offset residual negative impacts on biodiversity mentioned?"
                    ),
                    Question(
                        text="Are transformative actions and additional conservation actions described?"
                    ),
                ],
            ),
            Subcategory(
                id="biodiversity_planning",
                name="Consideration of Biodiversity in the Planning Stage",
                category=BiodiversityCategory.PLANNING,
                questions=[
                    Question(
                        text="Is biodiversity assessment or consideration explicitly mentioned in the planning stage?"
                    ),
                    Question(
                        text="Are results of biodiversity risk assessments provided?"
                    ),
                    Question(
                        text="Is stakeholder engagement during the planning process mentioned?"
                    ),
                ],
            ),
        ],
    ),
    # 2. Conservation and Restoration
    CategoryScore(
        category=BiodiversityCategory.CONSERVATION,
        subcategories=[
            Subcategory(
                id="extinct_species_conservation",
                name="Extinct Species Conservation Measures",
                category=BiodiversityCategory.CONSERVATION,
                questions=[
                    Question(
                        text="Are actions targeting endangered species mentioned?"
                    ),
                    Question(
                        text="Are specific measures to reduce species extinction risks outlined?"
                    ),
                    Question(
                        text="Are monitoring and management plans for extincted species mentioned?"
                    ),
                ],
            ),
            Subcategory(
                id="invasive_species_management",
                name="Invasive Alien Species Management",
                category=BiodiversityCategory.CONSERVATION,
                questions=[
                    Question(
                        text="Are measures to prevent the introduction of invasive alien species described?"
                    ),
                    Question(
                        text="Are specific actions to control the spread of invasive species detailed?"
                    ),
                    Question(
                        text="Are monitoring and management plans for invasive species included?"
                    ),
                ],
            ),
            Subcategory(
                id="pollution_control",
                name="Pollution Control Measures",
                category=BiodiversityCategory.CONSERVATION,
                questions=[
                    Question(
                        text="Are specific actions to reduce pollution impacts on biodiversity described?"
                    ),
                    Question(
                        text="Are pollution reduction technologies or strategies detailed?"
                    ),
                    Question(
                        text="Are pollution monitoring and reporting mechanisms outlined?"
                    ),
                ],
            ),
        ],
    ),
    # 3. Decision-Making and Disclosure
    CategoryScore(
        category=BiodiversityCategory.DECISION,
        subcategories=[
            Subcategory(
                id="biodiversity_decision_making",
                name="Biodiversity Integration in Decision-Making",
                category=BiodiversityCategory.DECISION,
                questions=[
                    Question(
                        text="Is biodiversity explicitly considered in corporate decision-making?"
                    ),
                    Question(
                        text="Are biodiversity-related policies or guidelines detailed?"
                    ),
                    Question(
                        text="Is biodiversity integration within supply chain management mentioned?"
                    ),
                ],
            ),
            Subcategory(
                id="business_disclosure",
                name="Business Disclosure",
                category=BiodiversityCategory.DECISION,
                questions=[
                    Question(
                        text="Is biodiversity-related risk considered to be or has been disclosed in the corporate? "
                    ),
                    Question(
                        text="Are biodiversity-related performance metrics or goals clearly mentioned?"
                    ),
                ],
            ),

             Subcategory(
                id="Financial_Aspects",
                name="Financial Aspects of Biodiversity",
                category=BiodiversityCategory.DECISION,
                questions=[
                    Question(
                        text="Are financial support or investments related to biodiversity mentioned? "
                    ),
                ],
            ),
        ],
    ),
    # 4. Technology, Knowledge, and Collaboration
    CategoryScore(
        category=BiodiversityCategory.TECHNOLOGY,
        subcategories=[
            Subcategory(
                id="capacity_building",
                name="Capacity-Building and Collaboration",
                category=BiodiversityCategory.TECHNOLOGY,
                questions=[
                    Question(
                        text="Are collaborations with other organizations for biodiversity conservation mentioned?"
                    ),
                    Question(
                        text="Are actions for technology transfer or knowledge sharing detailed?"
                    ),
                    Question(
                        text="Are training or capacity-building programs clearly described?"
                    ),
                ],
            ),
            Subcategory(
                id="knowledge_sharing",
                name="Knowledge Sharing and Accessibility",
                category=BiodiversityCategory.TECHNOLOGY,
                questions=[
                    Question(
                        text="Is biodiversity-related research or data sharing clearly mentioned?"
                    ),
                    Question(
                        text="Are open resources or tools supporting biodiversity actions detailed?"
                    ),
                ],
            ),
        ],
    ),
  
]


# Helper functions for text analysis
def analyze_text_content(
    text_content: str, report: BiodiversityReport, rag_system=None
) -> BiodiversityReport:
    """
    Analyze text content against the biodiversity framework using specialized agents

    This function implements analysis of text content against the biodiversity framework
    by leveraging the RAG system and specialized analysis agents.

    Args:
        text_content: Text to analyze
        report: Existing BiodiversityReport to update
        rag_system: Optional RAG system instance containing agents and vectorstore

    Returns:
        Updated BiodiversityReport with analysis results
    """

    # Process each category and subcategory in the report
    for i, category_score in enumerate(report.category_scores):
        print(f"Analyzing category: {category_score.category.value}")

        # Process each subcategory
        for j, subcategory in enumerate(category_score.subcategories):
            print(f"  Analyzing subcategory: {subcategory.name}")

            # Process each question in the subcategory
            for k, question in enumerate(subcategory.questions):
                print(f"    Processing question: {question.text}")

                try:
                    # Search for relevant content using vectorstore
                    relevant_docs = rag_system.vectorstore.similarity_search(
                        question.text, k=5
                    )

                    # Extract text from relevant documents
                    if relevant_docs:
                        # Add documents as excerpts
                        for doc in relevant_docs:
                            report.category_scores[i].subcategories[
                                j
                            ].add_relevant_excerpt(k, doc.page_content)

                        # Combine all relevant text
                        full_text = "\n".join(
                            [doc.page_content for doc in relevant_docs]
                        )

                        # Extract sentences to help with N validation
                        sentences = extract_sentences(full_text)
                        sentence_count = len(sentences)
                        print(f"      Text contains {sentence_count} sentences")

                        # Initialize insights dictionary
                        insights = {}

                        # Collect biodiversity analysis (combined sentiment and social media)
                        if "biodiversity" in rag_system.agents:
                            try:
                                # Check if this content contains social media content
                                tweet_docs = [
                                    doc
                                    for doc in relevant_docs
                                    if doc.metadata.get("content_type") == "tweet"
                                ]
                                
                                # Prepare analysis context
                                analysis_text = full_text
                                
                                # If we have both social media and forum data, combine them
                                if hasattr(rag_system, 'forum_context') and rag_system.forum_context:
                                    analysis_text += f"\n\nFORUM DISCUSSIONS:\n{rag_system.forum_context}"
                                    print(f"      Including forum discussions in analysis")

                                biodiversity_results = rag_system.agents["biodiversity"].analyze({"text": analysis_text})
                                if biodiversity_results:
                                    insights.update(biodiversity_results)
                                    print(f"      Biodiversity analysis complete")
                            except Exception as e:
                                print(f"Error in biodiversity analysis: {str(e)}")

                        # Legacy support - if old agents still exist
                        if "social_media" in rag_system.agents:
                            try:
                                # Check if this content contains social media content
                                tweet_docs = [
                                    doc
                                    for doc in relevant_docs
                                    if doc.metadata.get("content_type") == "tweet"
                                ]

                                has_social_content = len(tweet_docs) > 0

                                if has_social_content:
                                    # Create a more specialized context for tweets analysis
                                    # that preserves metadata and highlights the most relevant content
                                    tweet_context = ""

                                    # Add all tweets with their metadata for better context
                                    for idx, doc in enumerate(
                                        tweet_docs[:10]
                                    ):  # Limit to 10 most relevant tweets
                                        tweet_context += f"--- TWEET {idx+1} ---\n"
                                        tweet_context += doc.page_content + "\n"

                                        # Include additional metadata if available
                                        meta = doc.metadata
                                        if meta.get("author_description"):
                                            tweet_context += f"Author: {meta.get('author_description')}\n"
                                        if meta.get("author_location"):
                                            tweet_context += f"Location: {meta.get('author_location')}\n"
                                        if meta.get("date"):
                                            tweet_context += (
                                                f"Date: {meta.get('date')}\n"
                                            )

                                        tweet_context += "---\n\n"

                                    # Analyze the specialized tweet context
                                    social_results = rag_system.agents[
                                        "social_media"
                                    ].analyze({"text": tweet_context})

                                    if social_results:
                                        insights.update(social_results)
                                        print(
                                            f"      Social media analysis complete on {len(tweet_docs)} relevant tweets"
                                        )

                                        # Log key metrics for debugging
                                        if "social_media" in social_results:
                                            sm_data = social_results["social_media"]
                                            topics = ", ".join(
                                                sm_data.get("key_topics", [])
                                            )
                                            print(f"      → Topics: {topics}")
                                            print(
                                                f"      → Sentiment: {sm_data.get('sentiment_distribution', {})}"
                                            )
                                    else:
                                        print(
                                            "      Social media analysis returned no results"
                                        )
                            except Exception as e:
                                print(f"Error in social media analysis: {str(e)}")
                                import traceback

                                traceback.print_exc()

                        # Use the BiodiversityFrameworkAgent for N, S, M calculations
                        # Print the available agents for debugging
                        print(
                            f"      Available agents: {list(rag_system.agents.keys())}"
                        )

                        # Check if the framework agent exists
                        if "framework" in rag_system.agents:
                            try:
                                print(
                                    f"      Using BiodiversityFrameworkAgent for analysis"
                                )
                                framework_results = rag_system.agents[
                                    "framework"
                                ].analyze(
                                    {"text": full_text, "question": question.text}
                                )

                                if framework_results:
                                    # Validate N value to ensure it's realistic
                                    doc_count = len(relevant_docs)
                                    reported_n = framework_results["N"]

                                    # Apply sanity checks to N value based on sentence count
                                    if reported_n > sentence_count:
                                        # N cannot be higher than total sentence count
                                        adjusted_n = min(reported_n, sentence_count)
                                        print(
                                            f"      Warning: Adjusting N from {reported_n} to {adjusted_n} (total sentences: {sentence_count})"
                                        )
                                        framework_results["N"] = adjusted_n
                                    elif reported_n > sentence_count * 0.75:
                                        # If N is suspiciously high (>75% of sentences are relevant)
                                        print(
                                            f"      Note: N value {reported_n} is high relative to sentence count {sentence_count}"
                                        )

                                    # Log detailed information about the N calculation
                                    print(
                                        f"      Framework analysis: N={framework_results['N']} (from {sentence_count} sentences), S={framework_results['S']:.1f}, M={framework_results['M']:.1f}"
                                    )

                                    # Directly update question metrics using the validated framework agent results
                                    report.category_scores[i].subcategories[
                                        j
                                    ].questions[k].N = framework_results["N"]
                                    report.category_scores[i].subcategories[
                                        j
                                    ].questions[k].update_specificity(
                                        framework_results["S"]
                                    )
                                    report.category_scores[i].subcategories[
                                        j
                                    ].questions[k].update_multiplicity(
                                        framework_results["M"]
                                    )
                                else:
                                    print(
                                        f"      Framework agent returned no results, falling back to traditional method"
                                    )
                                    # Fall back to traditional method if framework analysis failed
                                    specificity = rag_system._calculate_specificity(
                                        full_text
                                    )
                                    multiplicity = rag_system._calculate_multiplicity(
                                        [doc.page_content for doc in relevant_docs]
                                    )

                                    # For N, use our more sophisticated estimation function
                                    estimated_n = estimate_relevant_mentions(
                                        full_text, question.text, sentence_count
                                    )
                                    report.category_scores[i].subcategories[
                                        j
                                    ].questions[k].N = estimated_n

                                    report.category_scores[i].subcategories[
                                        j
                                    ].questions[k].update_specificity(specificity)
                                    report.category_scores[i].subcategories[
                                        j
                                    ].questions[k].update_multiplicity(multiplicity)
                                    print(
                                        f"      Fallback analysis: N={estimated_n}, S={specificity:.1f}, M={multiplicity:.1f}"
                                    )
                            except Exception as e:
                                print(f"Error in framework analysis: {str(e)}")
                                print(f"      Exception details: {type(e).__name__}")
                                # Fall back to traditional method if framework agent fails
                                specificity = rag_system._calculate_specificity(
                                    full_text
                                )
                                multiplicity = rag_system._calculate_multiplicity(
                                    [doc.page_content for doc in relevant_docs]
                                )

                                # For N, use our more sophisticated estimation function
                                estimated_n = estimate_relevant_mentions(
                                    full_text, question.text, sentence_count
                                )
                                report.category_scores[i].subcategories[j].questions[
                                    k
                                ].N = estimated_n

                                report.category_scores[i].subcategories[j].questions[
                                    k
                                ].update_specificity(specificity)
                                report.category_scores[i].subcategories[j].questions[
                                    k
                                ].update_multiplicity(multiplicity)
                                print(
                                    f"      Fallback analysis: N={estimated_n}, S={specificity:.1f}, M={multiplicity:.1f}"
                                )
                        else:
                            print(
                                f"      WARNING: BiodiversityFrameworkAgent not found in agents dictionary!"
                            )
                            # Fall back to traditional method if framework agent not available
                            # Calculate specificity (S) - how specific the information is (0-100)
                            specificity = rag_system._calculate_specificity(full_text)

                            # Calculate multiplicity (M) - percent of redundant information (0-100)
                            multiplicity = rag_system._calculate_multiplicity(
                                [doc.page_content for doc in relevant_docs]
                            )

                            # For N, use our more sophisticated estimation function
                            estimated_n = estimate_relevant_mentions(
                                full_text, question.text, sentence_count
                            )

                            # Update question-level scoring metrics
                            report.category_scores[i].subcategories[j].questions[
                                k
                            ].N = estimated_n
                            report.category_scores[i].subcategories[j].questions[
                                k
                            ].update_specificity(specificity)
                            report.category_scores[i].subcategories[j].questions[
                                k
                            ].update_multiplicity(multiplicity)
                            print(
                                f"      Traditional analysis: N={estimated_n}, S={specificity:.1f}, M={multiplicity:.1f}"
                            )

                        # Update subcategory metrics from questions
                        report.category_scores[i].subcategories[
                            j
                        ].update_from_questions()

                        # Update category metrics from subcategories
                        report.category_scores[i].update_from_subcategories()

                        # Update report with social media insights if any were collected
                        if insights:
                            report.update_insights(subcategory.id, insights)
                    else:
                        print(f"    No relevant documents found for: {question.text}")

                except Exception as e:
                    print(f"Error processing question '{question.text}': {str(e)}")
                    continue

    # Normalize insights for the final report
    report.normalize_insights()

    # Final update of all category metrics
    for category_score in report.category_scores:
        category_score.update_from_subcategories()

    return report


def extract_sentences(text: str) -> List[str]:
    """
    Extract individual sentences from text to help with accurate N counting.

    Args:
        text: The text to extract sentences from

    Returns:
        A list of individual sentences
    """
    # Clean the text
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.strip()

    # Use NLTK to extract sentences
    sentences = nltk.sent_tokenize(text)

    # Post-process: filter out very short "sentences"
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    return sentences


def estimate_relevant_mentions(text: str, question: str, sentence_count: int) -> int:
    """
    Estimate the number of relevant mentions based on text, question and sentence count.

    Args:
        text: The text to analyze
        question: The question to check relevance against
        sentence_count: Total number of sentences

    Returns:
        Estimated number of relevant mentions
    """
    # Extract key terms from the question
    question_lower = question.lower()

    # Get the core concept of the question by removing common question prefixes
    prefixes = [
        "are there",
        "is there",
        "are",
        "is",
        "do",
        "does",
        "what",
        "when",
        "where",
        "how",
        "why",
        "which",
        "who",
        "can",
        "could",
        "should",
        "would",
        "will",
        "have",
        "has",
    ]

    for prefix in prefixes:
        if question_lower.startswith(prefix):
            core_question = question_lower[len(prefix) :].strip()
            break
    else:
        core_question = question_lower

    # Extract key terms from core question
    key_terms = [
        term.strip()
        for term in core_question.split()
        if len(term.strip()) > 3
        and term.lower()
        not in [
            "there",
            "clear",
            "clearly",
            "about",
            "what",
            "when",
            "where",
            "their",
            "they",
            "them",
            "with",
            "from",
            "into",
        ]
    ]

    # Calculate relevance based on key term occurrence
    text_lower = text.lower()
    term_count = sum(text_lower.count(term.lower()) for term in key_terms if term)

    # If very few matches, assume low relevance
    if term_count < 2:
        return max(0, min(2, sentence_count // 20))

    # If many matches but few sentences, scale appropriately
    if term_count > sentence_count:
        return max(1, min(sentence_count // 5, 10))

    # For typical cases, estimate based on term density relative to sentence count
    relevance_ratio = term_count / max(1, sentence_count)

    # Apply some heuristics for typical documents
    estimated_n = max(
        1, min(int(sentence_count * relevance_ratio * 0.5), int(sentence_count * 0.3))
    )

    # Cap at reasonable values
    return max(1, min(estimated_n, 15))
