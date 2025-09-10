from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Set, Union
from enum import Enum
import csv
from io import StringIO
import re
import nltk
import logging

# Configure NLTK data path and downloads in a deterministic way
def _ensure_nltk_data():
    """Ensure NLTK data is available with proper error handling."""
    try:
        # Try to use existing data first
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        # Only download if data is missing
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"NLTK download failed: {e}. Sentence tokenization may be limited.")

# Initialize NLTK data once
_ensure_nltk_data()


class BiodiversityCategory(str, Enum):
    PLANNING = "Planning and Management"
    CONSERVATION = "Conservation and Restoration"
    DECISION = "Decision-Making and Disclosure"
    TECHNOLOGY = "Technology, Knowledge, and Collaboration"
    FINANCIAL = "Financial Support and Investment"


class Question(BaseModel):
    text: str
    relevant_excerpts: List[str] = []
    # LLM-based scoring metrics (all in 0-1 range)
    N: float = 0.0  # Semantic relevance score (0-1): how well the question is answered
    S: float = 0.0  # Specificity score (0-1): how detailed/specific the answer is  
    M: float = 0.0  # Multiplicity/redundancy score (0-1): how much redundancy exists
    # Reasoning for scoring decisions
    scoring_rationale: str = ""

    def calculate_score(self) -> float:
        """Calculate question score using N × S × (1 − M) formula scaled to 1-10 range"""
        # Validate inputs are in 0-1 range (LLM should provide normalized values)
        if not (0 <= self.N <= 1) or not (0 <= self.S <= 1) or not (0 <= self.M <= 1):
            logging.warning(f"Invalid score metrics (should be 0-1): N={self.N}, S={self.S}, M={self.M}")
            return 0.0
            
        # Formula: N × S × (1 - M) gives a value in [0,1]
        # N: How well question is answered (0=not answered, 1=fully answered)
        # S: How specific/detailed the answer is (0=vague, 1=very specific)
        # M: Redundancy level (0=no redundancy, 1=highly redundant, so (1-M) rewards low redundancy)
        base_score = self.N * self.S * (1.0 - self.M)
        
        # Scale to 1-10 range: score = 1 + (base_score × 9)
        # This maps 0.0 -> 1.0 and 1.0 -> 10.0
        scaled_score = 1.0 + (base_score * 9.0)
        
        # Ensure score is bounded [1,10]
        return max(1.0, min(10.0, scaled_score))

    def add_relevant_excerpt(self, excerpt: str) -> None:
        """Add a relevant excerpt for reference"""
        if excerpt and excerpt.strip():
            self.relevant_excerpts.append(excerpt.strip())

    def update_llm_scores(self, n: float, s: float, m: float, rationale: str = "") -> None:
        """Update LLM-based scores with validation"""
        self.N = max(0.0, min(1.0, float(n)))
        self.S = max(0.0, min(1.0, float(s))) 
        self.M = max(0.0, min(1.0, float(m)))
        self.scoring_rationale = rationale or ""

    # Legacy methods for backward compatibility - convert from 0-100 to 0-1 range
    def update_specificity(self, value: float) -> None:
        """Legacy method - converts 0-100 range to 0-1"""
        self.S = max(0.0, min(1.0, float(value) / 100.0))

    def update_multiplicity(self, value: float) -> None:
        """Legacy method - converts 0-100 range to 0-1"""  
        self.M = max(0.0, min(1.0, float(value) / 100.0))


class Subcategory(BaseModel):
    id: str
    name: str
    questions: List[Question]
    category: BiodiversityCategory

    # Aggregated scoring metrics at subcategory level (calculated from questions)
    N: float = 0.0  # Average N across questions
    S: float = 0.0  # Average S across questions  
    M: float = 0.0  # Average M across questions
    
    # Dollar value allocation for this subcategory
    dollar_value: float = 0.0

    def calculate_score(self) -> float:
        """Calculate subcategory score as mean of question scores (excluding missing)"""
        if not self.questions:
            return 0.0
            
        # Get scores for questions that have been evaluated (N > 0 or explicit scoring)
        valid_scores = []
        for q in self.questions:
            score = q.calculate_score()
            # Include questions that have been scored (N > 0 indicates LLM evaluation)
            if q.N > 0 or score > 0:
                valid_scores.append(score)
        
        if not valid_scores:
            logging.debug(f"No valid question scores found for subcategory: {self.name}")
            return 0.0
            
        return sum(valid_scores) / len(valid_scores)

    def add_relevant_excerpt(self, question_index: int, excerpt: str) -> None:
        """Add a relevant excerpt to a specific question"""
        if 0 <= question_index < len(self.questions):
            self.questions[question_index].add_relevant_excerpt(excerpt)
            # Update aggregated metrics after adding excerpt
            self.update_from_questions()

    def update_from_questions(self) -> None:
        """Update subcategory aggregated metrics based on question metrics"""
        if not self.questions:
            self.N = self.S = self.M = 0.0
            return

        # Only aggregate from questions that have been evaluated (N > 0)
        evaluated_questions = [q for q in self.questions if q.N > 0]
        
        if not evaluated_questions:
            self.N = self.S = self.M = 0.0
            return
            
        # Simple average of N, S, M across evaluated questions
        self.N = sum(q.N for q in evaluated_questions) / len(evaluated_questions)
        self.S = sum(q.S for q in evaluated_questions) / len(evaluated_questions)  
        self.M = sum(q.M for q in evaluated_questions) / len(evaluated_questions)
        
        # Ensure values stay within 0-1 bounds
        self.N = max(0.0, min(1.0, self.N))
        self.S = max(0.0, min(1.0, self.S))
        self.M = max(0.0, min(1.0, self.M))

    def update_specificity(self, value: float) -> None:
        """Legacy method - converts 0-100 to 0-1 and updates all questions"""
        normalized_value = max(0.0, min(1.0, float(value) / 100.0))
        for question in self.questions:
            question.S = normalized_value
        self.update_from_questions()  # Recalculate aggregated values

    def update_multiplicity(self, value: float) -> None:
        """Legacy method - converts 0-100 to 0-1 and updates all questions"""
        normalized_value = max(0.0, min(1.0, float(value) / 100.0))
        for question in self.questions:
            question.M = normalized_value
        self.update_from_questions()  # Recalculate aggregated values


class CategoryScore(BaseModel):
    category: BiodiversityCategory
    subcategories: List[Subcategory]
    weight: float = 1.0  # Default weight for category

    # Category-level aggregated scoring metrics (calculated from subcategories)
    N: float = 0.0  # Average N across subcategories
    S: float = 0.0  # Average S across subcategories
    M: float = 0.0  # Average M across subcategories

    def get_total_score(self) -> float:
        """Calculate mean score across subcategories (excluding missing)"""
        return self.calculate_score()

    def calculate_score(self) -> float:
        """Calculate category score as mean of subcategory scores (excluding missing)"""
        if not self.subcategories:
            return 0.0
            
        # Get scores for subcategories that have been evaluated
        valid_scores = []
        for sc in self.subcategories:
            score = sc.calculate_score()
            # Include subcategories that have valid question evaluations
            if score > 0:  # Non-zero score indicates some questions were evaluated
                valid_scores.append(score)
        
        if not valid_scores:
            logging.debug(f"No valid subcategory scores found for category: {self.category.value}")
            return 0.0
            
        return sum(valid_scores) / len(valid_scores)

    def update_from_subcategories(self) -> None:
        """Update category aggregated metrics based on subcategory values"""
        if not self.subcategories:
            self.N = self.S = self.M = 0.0
            return

        # Only aggregate from subcategories that have been evaluated
        evaluated_subcategories = [sc for sc in self.subcategories if sc.N > 0]
        
        if not evaluated_subcategories:
            self.N = self.S = self.M = 0.0
            return
            
        # Simple average of N, S, M across evaluated subcategories
        self.N = sum(sc.N for sc in evaluated_subcategories) / len(evaluated_subcategories)
        self.S = sum(sc.S for sc in evaluated_subcategories) / len(evaluated_subcategories)
        self.M = sum(sc.M for sc in evaluated_subcategories) / len(evaluated_subcategories)
        
        # Ensure values stay within 0-1 bounds
        self.N = max(0.0, min(1.0, self.N))
        self.S = max(0.0, min(1.0, self.S))
        self.M = max(0.0, min(1.0, self.M))

    def get_total_dollar_value(self) -> float:
        """Calculate total dollar value for this category"""
        return sum(subcategory.dollar_value for subcategory in self.subcategories)


class BiodiversityReport(BaseModel):
    category_scores: List[CategoryScore]
    additional_insights: Dict[str, Any] = {
        "sentiment": {
            "overall_score": 0.0,
            "positive_aspects": [],
            "negative_aspects": [],
            "key_emotions": [],
        },
        "stakeholder": {
            "total_stakeholders": [],
            "engagement_level": 0.0,
            "main_concerns": [],
        },
        "biodiversity": {
            "species_count": 0,
            "habitat_types": [],
            "conservation_status": {},
            "impact_score": 0.0,
        },
        "social_media": {
            "engagement_score": 0.0,
            "sentiment_distribution": {},
            "key_topics": [],
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
        if not insights:
            return
            
        # Helper function to safely extend lists avoiding duplicates
        def safe_extend_unique(target_list, new_items):
            if isinstance(new_items, (list, tuple, set)):
                for item in new_items:
                    if item and item not in target_list:
                        target_list.append(item)

        # Aggregate sentiment insights
        if "sentiment_score" in insights:
            score = insights["sentiment_score"]
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                self.additional_insights["sentiment"]["overall_score"] += score
                
            safe_extend_unique(
                self.additional_insights["sentiment"]["positive_aspects"],
                insights.get("positive_aspects", [])
            )
            safe_extend_unique(
                self.additional_insights["sentiment"]["negative_aspects"],
                insights.get("negative_aspects", [])
            )
            safe_extend_unique(
                self.additional_insights["sentiment"]["key_emotions"],
                insights.get("key_emotions", [])
            )

        # Aggregate stakeholder insights
        if "stakeholder_count" in insights:
            safe_extend_unique(
                self.additional_insights["stakeholder"]["total_stakeholders"],
                insights.get("key_stakeholders", [])
            )
            engagement = insights.get("engagement_level", 0)
            if isinstance(engagement, (int, float)) and 0 <= engagement <= 1:
                self.additional_insights["stakeholder"]["engagement_level"] += engagement
            safe_extend_unique(
                self.additional_insights["stakeholder"]["main_concerns"],
                insights.get("main_concerns", [])
            )

        # Aggregate biodiversity insights
        if "species_count" in insights:
            species_count = insights["species_count"]
            if isinstance(species_count, int) and species_count >= 0:
                self.additional_insights["biodiversity"]["species_count"] = max(
                    self.additional_insights["biodiversity"]["species_count"],
                    species_count
                )
            safe_extend_unique(
                self.additional_insights["biodiversity"]["habitat_types"],
                insights.get("habitat_types", [])
            )

            # Merge conservation status dictionaries with validation
            conservation_status = insights.get("conservation_status", {})
            if isinstance(conservation_status, dict):
                for status, status_count in conservation_status.items():
                    if isinstance(status_count, (int, float)) and status_count >= 0:
                        current = self.additional_insights["biodiversity"]["conservation_status"].get(status, 0)
                        self.additional_insights["biodiversity"]["conservation_status"][status] = current + status_count

            impact_score = insights.get("impact_score", 0)
            if isinstance(impact_score, (int, float)) and impact_score >= 0:
                self.additional_insights["biodiversity"]["impact_score"] += impact_score

        # Aggregate social media insights
        if "engagement_score" in insights:
            engagement_score = insights["engagement_score"]
            if isinstance(engagement_score, (int, float)) and 0 <= engagement_score <= 1:
                self.additional_insights["social_media"]["engagement_score"] += engagement_score

            # Merge sentiment distributions with validation
            sentiment_dist = insights.get("sentiment_distribution", {})
            if isinstance(sentiment_dist, dict):
                for sentiment, value in sentiment_dist.items():
                    if isinstance(value, (int, float)) and value >= 0:
                        current = self.additional_insights["social_media"]["sentiment_distribution"].get(sentiment, 0)
                        self.additional_insights["social_media"]["sentiment_distribution"][sentiment] = current + value

            safe_extend_unique(
                self.additional_insights["social_media"]["key_topics"],
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

        # Ensure all list values are properly formatted (already lists, no conversion needed)

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
    if not text or not isinstance(text, str):
        return []
        
    # Clean the text
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace
    text = text.strip()
    
    if not text:
        return []

    try:
        # Use NLTK to extract sentences with fallback
        sentences = nltk.sent_tokenize(text)
    except (LookupError, Exception):
        # Fallback to simple regex-based sentence splitting
        sentences = re.split(r'[.!?]+', text)

    # Post-process: filter out very short "sentences" and clean up
    sentences = [s.strip() for s in sentences 
                if s.strip() and len(s.strip()) > 10 and not s.strip().isdigit()]

    return sentences


def estimate_relevant_mentions(text: str, question: str, sentence_count: int) -> int:
    """
    Estimate the number of relevant mentions based on text, question and sentence count.
    Uses improved logic to avoid overestimation and provides more accurate N values.

    Args:
        text: The text to analyze
        question: The question to check relevance against
        sentence_count: Total number of sentences

    Returns:
        Estimated number of relevant mentions (0-15 range)
    """
    if not text or not question or sentence_count <= 0:
        return 0
        
    # Input validation
    if not isinstance(text, str) or not isinstance(question, str):
        return 0
        
    # Extract key terms from the question more intelligently
    question_lower = question.lower().strip()

    # Remove common question words and get meaningful terms
    stop_words = {
        "are", "is", "there", "clear", "clearly", "mentioned", "outlined", 
        "described", "detailed", "included", "specific", "actions", "measures",
        "plans", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "as", "that", "what", "when", "where", "how", "why", "which",
        "who", "can", "could", "should", "would", "will", "have", "has", "been",
        "their", "they", "them", "into", "about", "during"
    }
    
    # Extract meaningful terms (length > 3, not stop words)
    words = re.findall(r'\b\w+\b', question_lower)
    key_terms = [word for word in words 
                if len(word) > 3 and word not in stop_words]
    
    if not key_terms:
        return 0

    # Count occurrences in text with better matching
    text_lower = text.lower()
    
    # Use word boundaries to avoid partial matches
    term_matches = 0
    for term in key_terms:
        # Count exact word matches, not substring matches
        pattern = r'\b' + re.escape(term) + r'\b'
        matches = len(re.findall(pattern, text_lower))
        term_matches += min(matches, 5)  # Cap per term to avoid overweighting

    # No matches means no relevance
    if term_matches == 0:
        return 0

    # Conservative estimation based on sentence count and matches
    # Generally, relevant mentions should be much less than total sentences
    max_possible = min(sentence_count // 3, 10)  # At most 1/3 of sentences, cap at 10
    
    # Scale based on term matches but be conservative
    if term_matches <= 2:
        estimated = 1
    elif term_matches <= 5:
        estimated = min(2, max_possible)
    elif term_matches <= 10:
        estimated = min(3, max_possible)
    else:
        # Many matches - likely relevant document
        estimated = min(max(4, sentence_count // 8), max_possible)

    # Final bounds check
    return max(0, min(estimated, min(15, sentence_count)))
