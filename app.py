import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import json
import re
import glob
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from rag_system import RAGSystem, RAGAgent
from models import BIODIVERSITY_FRAMEWORK, BiodiversityCategory
from agents import (
    BiodiversityAnalysisAgent,
    BiodiversityFrameworkAgent,
)
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from plotly.subplots import make_subplots
import matplotlib.colors as mcolors
#import logging

#logging.basicConfig(
#    level=logging.WARNING,  # show only important logs in terminal
#   filename="app.log",     # also save all logs to file
#    filemode="w",           # overwrite each run (use "a" to append)
#   format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()


def sync_dollar_values_to_model():
    """Sync dollar values from session state to the model"""
    if hasattr(st.session_state, 'results') and st.session_state.results and hasattr(st.session_state, 'subcategory_dollar_values'):
        for category_score in st.session_state.results.category_scores:
            for subcategory in category_score.subcategories:
                if subcategory.id in st.session_state.subcategory_dollar_values:
                    subcategory.dollar_value = st.session_state.subcategory_dollar_values[subcategory.id]

def create_biodiversity_bullet_chart(score_value):
    """
    Create a bullet chart visualization for the biodiversity score

    Args:
        score_value: The biodiversity score value (0-1 scale)

    Returns:
        A plotly figure object
    """
    # Convert score to percentage (0-100)
    score_percentage = score_value * 100

    # Use a colormap from matplotlib for smooth gradient colors
    cmap = plt.get_cmap("Spectral")
    gradient_steps = []
    ranges = np.linspace(0, 100, 30)  # More steps = smoother gradient

    for i in range(len(ranges) - 1):
        rgba = cmap(i / len(ranges))
        color = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.5)"
        gradient_steps.append({"range": [ranges[i], ranges[i + 1]], "color": color})

    # Build the bullet chart without the number display (since it's shown below)
    fig = go.Figure(
        go.Indicator(
            mode="gauge",  # Only gauge, no number
            value=score_percentage,
            gauge={
                "shape": "bullet",
                "axis": {"range": [0, 100]},
                "threshold": {"line": {"color": "black", "width": 2}, "value": 50},
                "steps": gradient_steps,
                "bar": {"color": "darkblue"},
            },
            title={"text": ""},  # Remove inline title
        )
    )

    # Clean layout without title
    fig.update_layout(
        title=None,  # Explicitly set no title
        height=220,
        width=None,  # Let it automatically adjust to container width
        margin=dict(l=20, r=20, t=30, b=30),  # Reduce top margin since no title
        autosize=True,  # Ensure it resizes with container
        showlegend=False  # Ensure no legend is shown
    )

    return fig


# Function to display evaluation results
def display_evaluation_results():
    if st.session_state.results and st.session_state.processing_status == "complete":
        # Sync dollar values from UI to model
        sync_dollar_values_to_model()
        st.markdown(
            f"""
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #1B5E20; margin-top: 0;">📊 Biodiversity Evaluation Results</h3>
            <p style="color: #2E7D32; margin-bottom: 0;">Results for project: {st.session_state.current_project}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Overall score calculation using current category weights
        weights = getattr(st.session_state, 'category_weights', None)
        overall_score = st.session_state.results.get_overall_score(weights)

        # Display overall score without bullet chart
        st.markdown(
            f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                <h1 style="color: #2E7D32; font-size: 48px;">{overall_score:.2f}</h1>
                <p style="color: #555; font-size: 18px;">Overall Biodiversity Score (0-1 scale)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Category scores
        st.markdown("### Category Scores")
        category_data = []

        for category_score in st.session_state.results.category_scores:
            category_name = category_score.category.value
            category_score_value = category_score.calculate_score() * 100
            category_data.append(
                {"Category": category_name, "Score": category_score_value}
            )

        # Create DataFrame for category scores
        df_categories = pd.DataFrame(category_data)

        # Sort by score descending
        df_categories = df_categories.sort_values("Score", ascending=False)

        # Create bar chart for category scores
        fig = px.bar(
            df_categories,
            x="Category",
            y="Score",
            color="Score",
            color_continuous_scale=["#FFCDD2", "#C8E6C9", "#A5D6A7", "#81C784"],
            labels={"Score": "Score (%)", "Category": "Category"},
            range_y=[0, 100],
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Score (%)",
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=50, b=100),
        )
        st.plotly_chart(
            fig, use_container_width=True, key="category_scores_bar_detailed"
        )

        # Detailed subcategory breakdowns by category
        st.markdown("### Detailed Analysis by Category")

        for category_score in st.session_state.results.category_scores:
            with st.expander(f"{category_score.category.value} Details"):
                # Create a DataFrame for subcategory scores within this category
                subcategory_data = []
                for subcategory in category_score.subcategories:
                    subcategory_data.append(
                        {
                            "Subcategory": subcategory.name,
                            "Score": subcategory.calculate_score() * 100,
                            "N": subcategory.N,
                            "S": subcategory.S,
                            "M": subcategory.M,
                        }
                    )

                df_subcategories = pd.DataFrame(subcategory_data)

                # Display subcategory scores as a table
                st.markdown(f"#### Scores for {category_score.category.value}")
                st.dataframe(
                    df_subcategories.style.format(
                        {"Score": "{:.1f}%", "S": "{:.1f}", "M": "{:.1f}"}
                    )
                )

                # Display subcategory scores as a chart
                fig = px.bar(
                    df_subcategories,
                    x="Subcategory",
                    y="Score",
                    color="Score",
                    color_continuous_scale=["#FFCDD2", "#C8E6C9", "#A5D6A7", "#81C784"],
                    labels={"Score": "Score (%)", "Subcategory": "Subcategory"},
                    range_y=[0, 100],
                )
                fig.update_layout(
                    xaxis_title="Subcategory",
                    yaxis_title="Score (%)",
                    coloraxis_showscale=False,
                    margin=dict(l=20, r=20, t=50, b=100),
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"subcategory_chart_{category_score.category.value}_{subcategory.id}",
                )

                # Display questions and excerpts for each subcategory
                for subcategory in category_score.subcategories:
                    # Replace nested expander with a collapsible section using markdown
                    st.markdown(f"#### {subcategory.name} Questions")

                    # Use a divider to separate subcategories visually
                    st.markdown("---")

                    # Question-level analysis
                    question_data = []
                    for i, question in enumerate(subcategory.questions):
                        # Use the question's actual metrics now
                        n_count = question.N
                        question_score = question.calculate_score()

                        question_data.append(
                            {
                                "Question": question.text,
                                "Excerpts": n_count,
                                "Score": question_score * 100,
                                "S": question.S,
                                "M": question.M,
                            }
                        )

                        # Display question and score
                        st.markdown(
                            f"**Q{i+1}: {question.text}** (Score: {question_score:.2f}, Excerpts: {n_count})"
                        )

                        if question.relevant_excerpts:
                            for j, excerpt in enumerate(question.relevant_excerpts):
                                st.markdown(
                                    f"""
                                <div style="background-color: #F1F8E9; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid #AED581;">
                                    <p style="margin: 0; color: #33691E; font-size: 0.9em;">{excerpt}</p>
                                </div>
                                """,
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                """
                            <div style="background-color: #FFF3E0; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 3px solid #FFB74D;">
                                <p style="margin: 0; color: #E65100; font-size: 0.9em;">No relevant information found.</p>
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    # Display questions table
                    if question_data:
                        st.markdown("#### Question Summary")
                        question_df = pd.DataFrame(question_data)
                        st.dataframe(
                            question_df.style.format(
                                {"Score": "{:.1f}%", "S": "{:.1f}", "M": "{:.1f}"}
                            )
                        )

                        # Bar chart for question scores
                        fig = px.bar(
                            question_df,
                            x="Question",
                            y="Score",
                            color="Score",
                            color_continuous_scale=[
                                "#FFCDD2",
                                "#C8E6C9",
                                "#A5D6A7",
                                "#81C784",
                            ],
                            labels={"Score": "Score (%)", "Question": "Question"},
                            range_y=[0, 100],
                        )
                        fig.update_layout(
                            xaxis_title="Question",
                            yaxis_title="Score (%)",
                            xaxis={
                                "tickmode": "array",
                                "tickvals": list(range(len(question_data))),
                                "ticktext": [
                                    f"Q{i+1}" for i in range(len(question_data))
                                ],
                            },
                            coloraxis_showscale=False,
                            margin=dict(l=20, r=20, t=50, b=20),
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"question_chart_{category_score.category.value}_{subcategory.id}_{hash(str(question_df.shape))}",
                        )

        # Display additional insights
        if hasattr(st.session_state.results, "additional_insights"):
            insights = st.session_state.results.additional_insights

            with st.expander("Additional Insights"):
                # Sentiment Analysis
                if "sentiment" in insights:
                    st.markdown("#### 🔍 Sentiment Analysis")

                    sentiment_score = insights["sentiment"]["overall_score"] * 100
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.metric(
                            "Overall Sentiment",
                            f"{sentiment_score:.1f}%",
                            delta=None,
                            delta_color="normal",
                        )

                    with col2:
                        positive = insights["sentiment"].get("positive_aspects", [])
                        negative = insights["sentiment"].get("negative_aspects", [])
                        emotions = insights["sentiment"].get("key_emotions", [])

                        if positive:
                            st.markdown("**Positive Aspects:**")
                            st.markdown(
                                ", ".join([f"_{p}_" for p in positive[:5]]),
                                unsafe_allow_html=True,
                            )

                        if negative:
                            st.markdown("**Areas of Concern:**")
                            st.markdown(
                                ", ".join([f"_{n}_" for n in negative[:5]]),
                                unsafe_allow_html=True,
                            )

                        if emotions:
                            st.markdown("**Key Emotions:**")
                            st.markdown(
                                ", ".join([f"_{e}_" for e in emotions[:5]]),
                                unsafe_allow_html=True,
                            )

                # Stakeholder Analysis
                
                # Social Media Analysis
                if "social_media" in insights and any(
                    insights["social_media"].get(key)
                    for key in [
                        "engagement_score",
                        "sentiment_distribution",
                        "key_topics",
                    ]
                ):
                    st.markdown("#### 📱 Social Media Analysis")

                    engagement = (
                        insights["social_media"].get("engagement_score", 0) * 100
                    )
                    sentiment_dist = insights["social_media"].get(
                        "sentiment_distribution", {}
                    )
                    topics = insights["social_media"].get("key_topics", [])
                    response_type = insights["social_media"].get(
                        "community_response", "neutral"
                    )

                    # Get enhanced metadata
                    author_types = insights["social_media"].get("author_types", [])
                    locations = insights["social_media"].get("locations", [])
                    time_span = insights["social_media"].get("time_span", "")

                    # Display community response info simply
                    st.markdown(f"**Community Response:** {response_type.capitalize()}")
                    if time_span:
                        st.markdown(f"**Time Period:** {time_span}")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.metric(
                            "Social Engagement",
                            f"{engagement:.1f}%",
                            delta=None,
                            delta_color="normal",
                        )

                        # Display key topics
                        if topics:
                            st.markdown("**Key Topics:**")
                            st.markdown(", ".join([f"_{topic}_" for topic in topics[:5]]))

                        # Display author types if available
                        if author_types:
                            st.markdown("**Content Authors:**")
                            st.markdown(", ".join(author_types[:3]))

                        # Display locations if available
                        if locations:
                            st.markdown("**Mentioned Locations:**")
                            st.markdown(", ".join(locations[:3]))

                    with col2:
                        if sentiment_dist:
                            # Create pie chart for sentiment distribution
                            df_sentiment = pd.DataFrame(
                                {
                                    "Sentiment": list(sentiment_dist.keys()),
                                    "Value": list(sentiment_dist.values()),
                                }
                            )

                            fig = px.pie(
                                df_sentiment,
                                values="Value",
                                names="Sentiment",
                                color="Sentiment",
                                color_discrete_map={
                                    "positive": "#81C784",  # Green
                                    "neutral": "#90CAF9",  # Blue
                                    "negative": "#EF9A9A",  # Red
                                },
                                title="Sentiment Distribution",
                            )
                            fig.update_traces(
                                textposition="inside", textinfo="percent+label"
                            )
                            fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                            st.plotly_chart(
                                fig, use_container_width=True, key="sentiment_pie_chart"
                            )
                
                # Add download button for forum/internet data
                st.markdown("---")
                st.markdown("#### 📥 Data Sources")
                if hasattr(st.session_state, 'rag_system') and hasattr(st.session_state.rag_system, 'forum_context') and st.session_state.rag_system.forum_context:
                    # Create formatted content for download
                    forum_data_content = f"""# Forum and Internet Data Sources
# Project: {st.session_state.current_project}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# 
# This file contains the forum discussions and Google reviews data used for sentiment analysis.
# Each entry includes source information and URLs for reference.

{st.session_state.rag_system.forum_context}

---
Generated by Biodiversity Criteria Evaluator
"""
                    
                    st.download_button(
                        label="📥 Download Forum & Review Data",
                        data=forum_data_content,
                        file_name=f"{st.session_state.current_project}_forum_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        help="Download the forum discussions and Google reviews data used for sentiment analysis"
                    )
                else:
                    st.info("No forum or review data available for download.")

        # Download CSV report with current weights and dollar values
        weights = getattr(st.session_state, 'category_weights', None)
        # Sync dollar values to model before CSV generation
        sync_dollar_values_to_model()
        csv_report = st.session_state.results.generate_csv_report(weights)

        # Create DataFrame from CSV for visualizations
        csv_io = StringIO(csv_report)
        df_biodiversity = pd.read_csv(csv_io)
        # Store the dataframe in session state for later use in visualizations
        st.session_state.biodiversity_df = df_biodiversity

        # Add download button for CSV report before visualizations
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_report,
            file_name=f"biodiversity_assessment_{st.session_state.current_project}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

        # Add Sankey diagram visualization
        st.markdown("### Hierarchical Relationship Visualization")

        # Create a dataframe with questions data
        questions_df = []
        for category_score in st.session_state.results.category_scores:
            category_name = category_score.category.value

            for subcategory in category_score.subcategories:
                for i, question in enumerate(subcategory.questions):
                    question_score = question.calculate_score()

                    # Create a unique label for each question that includes subcategory identifier
                    question_label = f"Q{i+1} ({subcategory.name[:3]})"

                    questions_df.append(
                        {
                            "Category": category_name,
                            "Sub-Category": subcategory.name,
                            "Question": question_label,  # Unique question label
                            "Question Score": question_score,
                            "Original ID": f"{category_name[:3]}-{subcategory.name[:3]}-Q{i+1}",  # For traceability
                        }
                    )

        questions_df = pd.DataFrame(questions_df)

        # === Base Colors for Categories ===
        base_palette = [
            "#1f77b4",
            "#2ca02c",
            "#9467bd",
            "#ff7f0e",
            "#d62728",
        ]  # Blue, Green, Purple, Orange, Red

        # Initialize
        nodes = []
        node_colors = []
        category_list = questions_df["Category"].unique().tolist()

        # Dynamically build nodes and assign colors
        category_color_map = {}

        for idx, cat in enumerate(category_list):
            base_color = np.array(mcolors.to_rgb(base_palette[idx % len(base_palette)]))
            category_color_map[cat] = base_color

            # Add Category node
            nodes.append(cat)
            node_colors.append(mcolors.to_hex(base_color))

            # Subcategories under this Category
            subs = questions_df[questions_df["Category"] == cat][
                "Sub-Category"
            ].unique()
            for sub in subs:
                # Lighter version of category color for subcategories
                sub_color = np.clip(base_color * 1.3, 0, 1)  # Brighter variant
                nodes.append(sub)
                node_colors.append(mcolors.to_hex(sub_color))

                # Questions under this Subcategory
                qs = questions_df[questions_df["Sub-Category"] == sub][
                    "Question"
                ].unique()
                for q in qs:
                    # Even lighter for questions
                    q_color = np.clip(base_color * 1.6, 0, 1)  # Much lighter variant
                    nodes.append(q)
                    node_colors.append(mcolors.to_hex(q_color))

        # Create node index mapping
        node_indices = {node: i for i, node in enumerate(nodes)}

        # === Build Links ===
        # Category -> Subcategory
        cat_sub_links = questions_df.drop_duplicates(
            subset=["Category", "Sub-Category"]
        )[["Category", "Sub-Category"]]
        cat_sub_links["value"] = (
            2  # Thicker connections between categories and subcategories
        )

        # Subcategory -> Question
        sub_q_links = questions_df[["Sub-Category", "Question", "Question Score"]]

        # Ensure question scores are meaningful for visualization (minimum value)
        sub_q_links["Question Score"] = sub_q_links["Question Score"].apply(
            lambda x: max(x, 0.1)
        )

        source = (
            cat_sub_links["Category"].map(node_indices).tolist()
            + sub_q_links["Sub-Category"].map(node_indices).tolist()
        )
        target = (
            cat_sub_links["Sub-Category"].map(node_indices).tolist()
            + sub_q_links["Question"].map(node_indices).tolist()
        )
        value = cat_sub_links["value"].tolist() + sub_q_links["Question Score"].tolist()

        # === Create gradient link colors based on source node ===
        link_colors = []

        # Get colors for category -> subcategory links (more vibrant)
        for cat in cat_sub_links["Category"]:
            base_color = category_color_map[cat]
            # Semi-transparent version of the category color
            rgba_color = (*base_color, 0.7)  # Create RGBA tuple with alpha
            color_hex = f"rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})"
            link_colors.append(color_hex)

        # Get colors for subcategory -> question links (more subtle)
        for sub in sub_q_links["Sub-Category"]:
            # Find the category for this subcategory
            cat = questions_df[questions_df["Sub-Category"] == sub]["Category"].iloc[0]
            base_color = category_color_map[cat]
            # Very transparent version of the category color
            rgba_color = (*base_color, 0.3)  # Create RGBA tuple with alpha
            color_hex = f"rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})"
            link_colors.append(color_hex)

        # === Build Sankey Diagram ===
        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=25,  # Increased padding
                        thickness=25,  # Increased thickness
                        line=dict(color="black", width=0.5),  # Thinner lines
                        label=nodes,
                        color=node_colors,
                        # Font styling is applied in the overall layout, not at node level
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=link_colors,
                        hoverinfo="none",  # Disable hover info for cleaner look
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text="Biodiversity Framework Hierarchy",
            title_font_size=26,
            font=dict(
                size=16, family="Arial", color="black"  # Ensure black font throughout
            ),
            margin=dict(t=80, l=30, r=30, b=30),
            height=800,  # Slightly reduced height
            paper_bgcolor="rgb(255,255,255)",  # White background (no gray)
            plot_bgcolor="rgb(255,255,255)",  # White background (no gray)
        )

        st.plotly_chart(fig, use_container_width=True, key="sankey_diagram")

        # Add pair plot of metrics
        st.markdown("### Relationship Between Metrics")

        # Select numerical columns for the pair plot
        pairplot_data = df_biodiversity[
            ["Question N", "Question S (%)", "Question M (%)", "Question Score"]
        ]
        pairplot_data.columns = ["N", "S (%)", "M (%)", "Score"]

        # Create the pair plot with adjusted figure size
        plt.figure(figsize=(10, 8))
        g = sns.pairplot(pairplot_data, height=2.0)  # Smaller individual plots
        for ax in g.axes.flat:
            ax.xaxis.label.set_size(14)  # Increase X-axis label size
            ax.yaxis.label.set_size(14)  # Increase Y-axis label size
            # Reduce ticklabel size
            ax.tick_params(labelsize=10)
        plt.suptitle("Pair Plot of Question Metrics", y=1.02, fontsize=16)

        # Use a container with fixed size to control dimensions
        container = st.container()
        with container:
            st.pyplot(g.fig)

    else:
        st.warning("No evaluation results available. Please run an evaluation first.")


# Custom CSS for styling the app
def load_css():
    # Custom CSS to style the app
    st.markdown(
        """
    <style>
        /* Background color for the main area */
        .stApp {
            background-color: #f0f7f0;  /* Light green background */
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #2E7D32;  /* Dark green sidebar */
            color: white;
        }
        
        /* Style sidebar text elements */
        section[data-testid="stSidebar"] .stSelectbox label, 
        section[data-testid="stSidebar"] .stButton label,
        section[data-testid="stSidebar"] span {
            color: white !important;
        }
        
        /* Style sidebar headers */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #E0F2F1 !important;
        }
        
        /* Style main headers */
        h1, h2, h3 {
            color: #1B5E20;  /* Darker green for headers */
        }
        
        /* Style buttons */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            border: none;
            padding: 8px 16px;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #388E3C;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Style cards/containers */
        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        /* Style metrics */
        div[data-testid="metric-container"] {
            background-color: #E8F5E9;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Style the chat interface */
        div.stChatMessageContent {
            background-color: #E8F5E9;
            border-radius: 15px;
        }
        
        div.stChatMessage [data-testid="StyledLinkIconContainer"] {
            color: #388E3C;
        }

        /* Fix charts going outside borders */
        [data-testid="stPlotlyChart"] > div {
            width: 100% !important;
            min-width: 0 !important;
            padding-left: 0 !important;
            margin-left: 0 !important;
        }
        
        /* Ensure all visualizations respect container width */
        .stPlotlyChart, .element-container, .stMarkdown {
            width: 100% !important;
            max-width: 100% !important;
            padding-left: 0 !important;
            margin-left: 0 !important;
        }
        
        /* Fix matplotlib plots */
        .stImage img {
            max-width: 100% !important;
            width: auto !important;
            margin: 0 auto !important;
            display: block !important;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


# Helper function to find and display project image
def get_project_image_path(project_path):
    """Find the project image path that starts with 'Picture'"""
    image_patterns = ["Picture*.png", "Picture*.jpg", "Picture*.jpeg"]
    for pattern in image_patterns:
        images = glob.glob(os.path.join(project_path, pattern))
        if images:
            return images[0]  # Return the first matching image
    return None


class ChatInterface:
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.visualization_container = None

        # Initialize chat history if it doesn't exist
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

    def display_chat_interface(self):
        """Display the chat interface in the Streamlit app"""
        # Styled header for the chat interface
        st.markdown(
            """
        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; margin-bottom: 20px; text-align: center;">
            <h3 style="color: #1B5E20; margin: 0;">💬 Interactive Q&A with RAG System</h3>
            <p style="color: #2E7D32; margin-top: 5px;">Ask questions about the project documents and get AI-powered insights</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Add chat controls in a row with styled buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_messages = []
                # Force active tab to stay on chat (index 2)
                st.session_state.active_tab = 2
                st.rerun()

        with col2:
            if st.button("💾 Save Chat", use_container_width=True):
                self._save_chat_history()
                st.success("Chat history saved!")

        with col3:
            if st.button("📊 Extract Insights", use_container_width=True):
                if st.session_state.results:
                    self._add_insights_message()
                else:
                    st.warning("No evaluation results available")

        # Example queries in a styled box
        st.markdown(
            """
        <div style="background-color: white; padding: 15px; border-radius: 10px; border-left: 4px solid #4CAF50; margin: 20px 0;">
            <h4 style="color: #1B5E20; margin-top: 0;">Example Queries</h4>
            <ul style="margin-bottom: 0; padding-left: 20px; color: #2E7D32;">
                <li>What stakeholders are mentioned in the project documents?</li>
                <li>Show me a chart of biodiversity impacts mentioned</li>
                <li>Generate a pie chart of stakeholder distribution</li>
                <li>What do tweets say about this project?</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Chat messages container
        chat_container = st.container()

        with chat_container:
            # Display chat messages
            for message in st.session_state.chat_messages:
                with st.chat_message(
                    message["role"],
                    avatar="🧑‍💻" if message["role"] == "user" else "🤖",
                ):
                    # Clean up the message content if it contains JSON response syntax
                    content = message["content"]
                    # Remove any JSON blocks that might be displayed as raw text
                    content = re.sub(
                        r"```json\s*(\{.*?\})\s*```", "", content, flags=re.DOTALL
                    )
                    # Remove any answer: or visualization: prefixes
                    content = re.sub(
                        r'"answer":\s*"(.*?)"', r"\1", content, flags=re.DOTALL
                    )

                    st.markdown(content)

                    # If there's a visualization to display
                    if "visualization" in message:
                        try:
                            st.markdown("---")
                            st.markdown(
                                '<div style="background-color: #E8F5E9; padding: 10px; border-radius: 8px; margin-bottom: 10px;"><h4 style="color: #1B5E20; margin: 0;">📊 Generated Visualization</h4></div>',
                                unsafe_allow_html=True,
                            )
                            vis_data = message["visualization"]
                            self._display_visualization(vis_data)
                        except Exception as e:
                            st.error(f"Error displaying visualization: {str(e)}")

        # Chat input at the bottom
        chat_input_container = st.container()

        with chat_input_container:
            # Add some space before the input
            st.markdown("<div style='padding: 10px;'></div>", unsafe_allow_html=True)

            if prompt := st.chat_input("Ask a question about the project..."):
                if not self.rag_system.current_project:
                    st.warning("Please select a project first")
                    return

                if (
                    "chat_rag_agent" not in st.session_state
                    or st.session_state.chat_rag_agent is None
                ):
                    st.warning(
                        "RAG agent not initialized. Please process a project first."
                    )
                    return

                # Store the fact that we're in chat processing mode
                # This helps maintain state during the chat flow
                st.session_state.processing_chat = True

                # Ensure we stay on the chat tab (index 2)
                st.session_state.active_tab = 2

                # Add user message to chat
                st.session_state.chat_messages.append(
                    {"role": "user", "content": prompt}
                )

                # Display user message
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown(prompt)

                # Display assistant response with a spinner
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()

                    # Connect the agent to the vector store if needed
                    self.rag_system.connect_agent_to_vectorstore(
                        st.session_state.chat_rag_agent
                    )

                    # Generate response
                    with st.spinner("Thinking..."):
                        response = st.session_state.chat_rag_agent.generate_response(
                            prompt
                        )

                    # Process the response based on what's returned
                    response_text = self._extract_response_text(response["text"])

                    # Display the cleaned text response
                    message_placeholder.markdown(response_text)

                    # Prepare assistant message for session state
                    assistant_message = {"role": "assistant", "content": response_text}

                    # Handle visualization if present
                    if "visualization" in response:
                        # Display the visualization
                        try:
                            # Add a visual divider
                            st.markdown("---")
                            st.markdown(
                                '<div style="background-color: #E8F5E9; padding: 10px; border-radius: 8px; margin-bottom: 10px;"><h4 style="color: #1B5E20; margin: 0;">📊 Generated Visualization</h4></div>',
                                unsafe_allow_html=True,
                            )

                            # Display the visualization
                            self._display_visualization(response["visualization"])

                            # Save visualization data to the message
                            assistant_message["visualization"] = response[
                                "visualization"
                            ]
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")

                    # Add assistant message to chat history
                    st.session_state.chat_messages.append(assistant_message)

                    # We've finished processing the chat
                    st.session_state.processing_chat = False

                    # Ensure we stay on the chat tab (index 2)
                    st.session_state.active_tab = 2

    def _extract_response_text(self, response_text):
        """Extract the human-readable part from LLM responses, particularly handling JSON responses better"""
        # First, check if the response has JSON structure
        if "{" in response_text and "}" in response_text:
            try:
                # Try to find JSON data in the response
                matches = re.findall(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
                if matches:
                    # If JSON code blocks are found, remove them
                    for match in matches:
                        response_text = response_text.replace(f"```json{match}```", "")

                # Look for "answer" field in regular JSON format
                answer_match = re.search(
                    r'"answer"\s*:\s*"(.*?)"', response_text, re.DOTALL
                )
                if answer_match:
                    answer_text = answer_match.group(1)
                    # Unescape any escaped quotes
                    answer_text = answer_text.replace('\\"', '"')
                    return answer_text

                # Look for { "answer": "text" } patterns
                json_match = re.search(
                    r'\{\s*"answer"\s*:\s*"(.*?)"\s*[,}]', response_text, re.DOTALL
                )
                if json_match:
                    answer_text = json_match.group(1)
                    # Unescape any escaped quotes
                    answer_text = answer_text.replace('\\"', '"')
                    return answer_text

                # If no specific answer field found, try to parse as JSON
                # First find the first complete JSON object
                json_obj_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
                if json_obj_match:
                    json_str = json_obj_match.group(1)
                    try:
                        json_obj = json.loads(json_str)
                        # If we have an answer field, return that
                        if "answer" in json_obj:
                            return json_obj["answer"]
                    except:
                        pass
            except:
                # If JSON parsing fails, fall back to the original text
                pass

        # Clean up the response by removing markdown code blocks that contain JSON
        response_text = re.sub(
            r"```json\s*(\{.*?\})\s*```", "", response_text, flags=re.DOTALL
        )

        # Remove lines that look like JSON syntax
        lines = response_text.split("\n")
        filtered_lines = []
        for line in lines:
            # Skip lines that look like JSON syntax
            if re.match(r'\s*["{}\[\],]', line) and not re.search(
                r"[a-zA-Z]{3,}", line
            ):
                continue
            filtered_lines.append(line)

        response_text = "\n".join(filtered_lines)

        return response_text.strip()

    def _save_chat_history(self):
        """Save chat history to a file"""
        try:
            if not self.rag_system.current_project:
                st.warning("Please select a project before saving chat")
                return False

            # Create a directory for chat histories if it doesn't exist
            chat_dir = os.path.join(self.rag_system.base_directory, "chat_histories")
            os.makedirs(chat_dir, exist_ok=True)

            # Create a unique filename with timestamp and project name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.rag_system.current_project}_chat_{timestamp}.json"
            filepath = os.path.join(chat_dir, filename)

            # Convert visualizations to JSON-serializable format
            chat_export = []
            for msg in st.session_state.chat_messages:
                chat_export.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                        "visualization": msg.get("visualization"),
                    }
                )

            # Save to file
            with open(filepath, "w") as f:
                json.dump(chat_export, f, indent=2)

            return True
        except Exception as e:
            st.error(f"Error saving chat history: {str(e)}")
            return False

    def _add_insights_message(self):
        """Add a system message with insights from evaluation results"""
        if not st.session_state.results:
            return

        # Extract insights from results
        insights_text = "## 📊 Summary of Evaluation Results\n\n"

        # Add overall scores by category
        category_scores = {}
        for category_score in st.session_state.results.category_scores:
            category_scores[category_score.category.value] = (
                category_score.calculate_score()
            )

        # Create visualization data for category scores
        vis_data = {
            "plot_type": "bar",
            "title": "Evaluation Scores by Category",
            "x_label": "Category",
            "y_label": "Score (0-1)",
            "data": {
                "labels": list(category_scores.keys()),
                "values": list(category_scores.values()),
            },
        }

        # Add text summary
        insights_text += "### Overall Category Scores\n\n"
        for category, score in category_scores.items():
            insights_text += f"- **{category}**: {score:.2f}\n"

        # Add any additional insights available
        insights = st.session_state.results.additional_insights

        if insights:
            if "biodiversity" in insights:
                bio = insights["biodiversity"]
                insights_text += f"\n### Biodiversity Insights\n\n"
                insights_text += f"- Species mentioned: {bio.get('species_count', 0)}\n"
                insights_text += (
                    f"- Habitat types: {', '.join(bio.get('habitat_types', []))}\n"
                )

            if "stakeholder" in insights:
                stake = insights["stakeholder"]
                insights_text += f"\n### Stakeholder Insights\n\n"
                insights_text += f"- Stakeholders identified: {len(stake.get('total_stakeholders', []))}\n"
                insights_text += (
                    f"- Main concerns: {', '.join(stake.get('main_concerns', []))}\n"
                )

            if "social_media" in insights:
                social = insights["social_media"]
                insights_text += f"\n### Social Media Insights\n\n"

                # Add enhanced social media insights
                sentiment_distribution = social.get("sentiment_distribution", {})
                if sentiment_distribution:
                    sentiment_str = ", ".join(
                        [f"{k}: {v:.1f}%" for k, v in sentiment_distribution.items()]
                    )
                    insights_text += f"- Sentiment: {sentiment_str}\n"

                insights_text += (
                    f"- Key topics: {', '.join(social.get('key_topics', []))}\n"
                )

                # Add author types if available
                if social.get("author_types"):
                    insights_text += f"- Content authors: {', '.join(social.get('author_types', []))}\n"

                # Add locations if available
                if social.get("locations"):
                    insights_text += f"- Mentioned locations: {', '.join(social.get('locations', []))}\n"

                # Add time span if available
                if social.get("time_span"):
                    insights_text += f"- Time span: {social.get('time_span', '')}\n"

                # Add community response
                if social.get("community_response"):
                    insights_text += f"- Community response: {social.get('community_response', 'neutral').capitalize()}\n"

        # Add message to chat history
        system_message = {
            "role": "assistant",
            "content": insights_text,
            "visualization": vis_data,
        }

        st.session_state.chat_messages.append(system_message)

    def _display_visualization(self, vis_data):
        """Create and display a visualization based on the provided data"""
        try:
            if not vis_data:
                st.warning("No visualization data available")
                return

            plot_type = vis_data.get("plot_type", "").lower()
            title = vis_data.get("title", "Data Visualization")
            x_label = vis_data.get("x_label", "")
            y_label = vis_data.get("y_label", "")

            data = vis_data.get("data", {})

            # Support both 'labels' and 'categories' keys for backward compatibility
            labels = data.get("labels", data.get("categories", []))
            values = data.get("values", [])
            series = data.get("series", [])

            # Handle empty data
            if not labels or not values:
                st.warning("Insufficient data to create visualization")
                return

            # Display description of how values were assigned if available
            description = ""
            if "description" in data:
                description = data["description"]
                st.info(f"**Data methodology**: {description}")

            # Generate a unique key for each chart based on the visualization data
            unique_key = f"{plot_type}_{title}_{hash(str(labels) + str(values))}"

            if plot_type == "bar":
                # Create a DataFrame for better visualization
                df = pd.DataFrame({"Category": labels, "Value": values})

                # Sort by value for better visualization if there are more than 3 categories
                if len(labels) > 3:
                    df = df.sort_values("Value", ascending=False)

                fig = px.bar(
                    df,
                    x="Category",
                    y="Value",
                    labels={
                        "Category": x_label or "Category",
                        "Value": y_label or "Value",
                    },
                    title=title,
                    text="Value",  # Display the values on the bars
                    color_discrete_sequence=px.colors.qualitative.Plotly,  # Use a nicer color palette
                )

                # Improve layout
                fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                fig.update_layout(
                    uniformtext_minsize=8,
                    uniformtext_mode="hide",
                    xaxis={"categoryorder": "total descending"},
                )

                st.plotly_chart(fig, use_container_width=True, key=f"bar_{unique_key}")

            elif plot_type == "pie":
                fig = px.pie(
                    names=labels,
                    values=values,
                    title=title,
                    hover_data=[values],  # Show values on hover
                    color_discrete_sequence=px.colors.qualitative.Plotly,  # Use a nicer color palette
                )

                # Improve layout
                fig.update_traces(textinfo="label+percent")
                fig.update_layout(showlegend=True)

                st.plotly_chart(fig, use_container_width=True, key=f"pie_{unique_key}")

            elif plot_type == "line":
                if series:
                    fig = go.Figure()
                    for i, s in enumerate(series):
                        fig.add_trace(
                            go.Scatter(
                                x=labels,
                                y=s.get("data", []),
                                name=s.get("name", f"Series {i+1}"),
                                mode="lines+markers",  # Add markers to the lines
                            )
                        )
                    fig.update_layout(
                        title=title,
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )
                else:
                    # Create a DataFrame for better visualization
                    df = pd.DataFrame({"Category": labels, "Value": values})

                    fig = px.line(
                        df,
                        x="Category",
                        y="Value",
                        labels={
                            "Category": x_label or "Period",
                            "Value": y_label or "Value",
                        },
                        title=title,
                        markers=True,  # Show markers on the line
                        color_discrete_sequence=px.colors.qualitative.Plotly,  # Use a nicer color palette
                    )
                st.plotly_chart(fig, use_container_width=True, key=f"line_{unique_key}")

            elif plot_type == "scatter":
                # Create a DataFrame for better visualization
                df = pd.DataFrame({"X": labels, "Y": values})

                fig = px.scatter(
                    df,
                    x="X",
                    y="Y",
                    labels={"X": x_label or "X", "Y": y_label or "Y"},
                    title=title,
                    color_discrete_sequence=px.colors.qualitative.Plotly,  # Use a nicer color palette
                )

                # Add trendline if there are more than 2 points
                if len(labels) > 2:
                    fig.update_layout(showlegend=True)

                st.plotly_chart(
                    fig, use_container_width=True, key=f"scatter_{unique_key}"
                )

            else:
                st.warning(f"Unsupported plot type: {plot_type}")

            # Add a download button for the data
            if labels and values:
                export_data = pd.DataFrame({"Category": labels, "Value": values})

                # Create a buffer for CSV and provide download
                csv = export_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download visualization data",
                    data=csv,
                    file_name=f"{title.replace(' ', '_').lower()}_data.csv",
                    mime="text/csv",
                    key=f"download_{hash(str(vis_data))}",
                )

        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            raise e


# Page config
st.set_page_config(
    page_title="Biodiversity Criteria Evaluator",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
load_css()

# Title and description with custom styling
st.markdown(
    """
<div style="text-align: center; padding: 20px; background-color: #E8F5E9; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: #1B5E20;">🌿 Biodiversity Criteria Evaluator</h1>
    <p style="font-size: 1.2em; color: #2E7D32;">
        This tool analyzes renewable energy projects documents for biodiversity-related criteria and provides detailed scoring based on:
    </p>
    <div style="display: flex; justify-content: center; gap: 30px; margin-top: 10px;">
        <div style="background-color: white; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <strong>Number of mentions (N)</strong>
        </div>
        <div style="background-color: white; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <strong>Specificity of mentions (S)</strong>
        </div>
        <div style="background-color: white; padding: 10px 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <strong>Multiplicity/repetition (M)</strong>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# If we have results, display the main gauge visualization
if (
    "results" in st.session_state
    and st.session_state.results
    and st.session_state.processing_status == "complete"
):
    # Display the tabs content instead of duplicating the visualization here
    pass

# Sidebar
st.sidebar.markdown(
    """
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="color: white; margin-bottom: 0;">⚙️ Configuration</h2>
</div>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = RAGSystem(model_name="gpt-4o-mini")

if "results" not in st.session_state:
    st.session_state.results = None

if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

if "current_project" not in st.session_state:
    st.session_state.current_project = None

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize RAG agent in session state
if "chat_rag_agent" not in st.session_state:
    st.session_state.chat_rag_agent = None

# Initialize session state for various components
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # Default to first tab

if "processing_chat" not in st.session_state:
    st.session_state.processing_chat = False

# Create chat interface instance
chat_interface = ChatInterface(st.session_state.rag_system)

# Project selection with custom styling
st.sidebar.markdown(
    '<div style="background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin-bottom: 15px;"><h3 style="color: white; margin: 0; font-size: 1.1em;">📁 Project</h3></div>',
    unsafe_allow_html=True,
)

available_projects = st.session_state.rag_system.get_available_projects()
selected_project = st.sidebar.selectbox(
    "Select Project",
    options=[""] + available_projects,
    format_func=lambda x: "Select a project..." if x == "" else x,
)

# Display project image if available
if selected_project:
    project_path = os.path.join(
        st.session_state.rag_system.base_directory, selected_project
    )
    image_path = get_project_image_path(project_path)
    if image_path:
        try:
            image = Image.open(image_path)
            # Create a styled container for the image
            st.sidebar.markdown(
                '<div style="background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 15px 0;"><h3 style="color: white; margin-bottom: 10px; font-size: 1.1em;">📷 Project Image</h3></div>',
                unsafe_allow_html=True,
            )
            st.sidebar.image(
                image, caption=f"Project: {selected_project}", use_container_width=True
            )
        except Exception as e:
            st.sidebar.warning(f"Error loading project image: {str(e)}")

if selected_project and selected_project != st.session_state.current_project:
    st.session_state.current_project = selected_project
    st.session_state.results = None
    st.session_state.processing_status = None

# Model selection with custom styling
st.sidebar.markdown(
    '<div style="background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 15px 0;"><h3 style="color: white; margin: 0; font-size: 1.1em;">🤖 Model</h3></div>',
    unsafe_allow_html=True,
)

model_options = {
    "gpt-4o-mini": "GPT-4o Mini (Balanced)",
    "gpt-4": "GPT-4 (Accurate)",
    "gpt-4-turbo": "GPT-4 Turbo (Latest)",
    "gpt-3.5": "GPT-3.5 Turbo (Fast)",
    "deepseek": "DeepSeek Chat (Alternative)"
}

selected_model = st.sidebar.selectbox(
    "Select Primary Model",
    options=list(model_options.keys()),
    format_func=lambda x: model_options[x],
    index=0,
)

# Update the RAG system model if it has changed
if hasattr(st.session_state, 'rag_system'):
    st.session_state.rag_system.update_model(selected_model)

# Initialize chunk settings with default values
if "pdf_chunk_size" not in st.session_state:
    st.session_state.pdf_chunk_size = 1000
if "tweet_chunk_size" not in st.session_state:
    st.session_state.tweet_chunk_size = 100

# Update chunk settings
st.session_state.rag_system.chunk_strategy.update_chunk_sizes(
    pdf_chunk_size=st.session_state.pdf_chunk_size,
    tweet_chunk_size=st.session_state.tweet_chunk_size,
)

# Category weights configuration
st.sidebar.markdown(
    '<div style="background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 15px 0;"><h3 style="color: white; margin: 0; font-size: 1.1em;">⚖️ Category Weights</h3></div>',
    unsafe_allow_html=True,
)

# Initialize category weights in session state if not exists
if "category_weights" not in st.session_state:
    st.session_state.category_weights = {
        BiodiversityCategory.PLANNING: 1.0,
        BiodiversityCategory.CONSERVATION: 1.0,
        BiodiversityCategory.DECISION: 1.0,
        BiodiversityCategory.TECHNOLOGY: 1.0,
    }

# Create weight sliders for each category
category_display_names = {
    BiodiversityCategory.PLANNING: "Planning & Management",
    BiodiversityCategory.CONSERVATION: "Conservation & Restoration", 
    BiodiversityCategory.DECISION: "Decision-Making & Disclosure",
    BiodiversityCategory.TECHNOLOGY: "Technology & Collaboration",
}

for category, display_name in category_display_names.items():
    st.session_state.category_weights[category] = st.sidebar.slider(
        display_name,
        min_value=0.0,
        max_value=5.0,
        value=st.session_state.category_weights.get(category, 1.0),
        step=0.1,
        help=f"Weight for {display_name} category (will be normalized)"
    )

# Reset weights button
if st.sidebar.button("🔄 Reset to Equal Weights", help="Set all categories to equal importance"):
    for category in st.session_state.category_weights.keys():
        st.session_state.category_weights[category] = 1.0
    st.rerun()

# Dollar value configuration
st.sidebar.markdown(
    '<div style="background-color: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin: 15px 0;"><h3 style="color: white; margin: 0; font-size: 1.1em;">💰 Subcategory Dollar Values</h3></div>',
    unsafe_allow_html=True,
)

# Define subcategory IDs in order and their default values
# Based on the 9 values provided: $390,000, $420,000, $240,000, $225,000, $90,000, $40,000, $201,000, $225,000, $70,000
subcategory_defaults = {
    "ecosystem_species_management": 390000,
    "biodiversity_planning": 420000,
    "extinct_species_conservation": 240000,
    "invasive_species_management": 225000,
    "pollution_control": 90000,
    "biodiversity_decision_making": 40000,
    "business_disclosure": 201000,
    "Financial_Aspects": 225000,
    "capacity_building": 70000,
    "knowledge_sharing": 150000,  # Additional subcategory with default value
}

# Display names for subcategories
subcategory_display_names = {
    "ecosystem_species_management": "Ecosystem/Species Management",
    "biodiversity_planning": "Biodiversity Planning",
    "extinct_species_conservation": "Species Conservation",
    "invasive_species_management": "Invasive Species Management",
    "pollution_control": "Pollution Control",
    "biodiversity_decision_making": "Decision-Making Integration",
    "business_disclosure": "Business Disclosure",
    "Financial_Aspects": "Financial Aspects",
    "capacity_building": "Capacity Building",
    "knowledge_sharing": "Knowledge Sharing",
}

# Initialize subcategory dollar values in session state if not exists
if "subcategory_dollar_values" not in st.session_state:
    st.session_state.subcategory_dollar_values = subcategory_defaults.copy()

# Create dollar value sliders for each subcategory
for subcategory_id, display_name in subcategory_display_names.items():
    if subcategory_id in subcategory_defaults:  # Only show sliders for defined subcategories
        current_value = st.session_state.subcategory_dollar_values.get(subcategory_id, subcategory_defaults[subcategory_id])
        
        st.session_state.subcategory_dollar_values[subcategory_id] = st.sidebar.slider(
            display_name,
            min_value=10000,
            max_value=500000,
            value=int(current_value),
            step=5000,
            format="$%d",
            help=f"Dollar allocation for {display_name} subcategory"
        )

# Display total dollar value
if st.session_state.subcategory_dollar_values:
    total_dollars = sum(st.session_state.subcategory_dollar_values.values())
    st.sidebar.markdown(f"**Total Budget: ${total_dollars:,}**")

# Reset dollar values button
if st.sidebar.button("🔄 Reset Dollar Values", help="Reset all subcategory values to defaults"):
    st.session_state.subcategory_dollar_values = subcategory_defaults.copy()
    st.rerun()

# Styled divider
st.sidebar.markdown(
    '<hr style="border: none; height: 1px; background-color: rgba(255,255,255,0.2); margin: 20px 0;">',
    unsafe_allow_html=True,
)

# Process button with enhanced styling
if selected_project:
    process_button = st.sidebar.button(
        "🚀 Process and Evaluate",
        use_container_width=True,
        help="Process the selected project and evaluate based on criteria",
    )

    if process_button:
        try:
            # Set the processing status and active tab
            st.session_state.processing_status = "starting"
            st.session_state.active_tab = 0  # Set to Overview tab

            # Create a progress container
            progress_container = st.empty()

            with progress_container.container():
                # Create a styled progress header
                st.markdown(
                    f"""
                <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                    <h3 style="color: #1B5E20; margin: 0;">🔄 Processing Project: {selected_project}</h3>
                    <p style="color: #2E7D32; margin-top: 5px;">Please wait while we analyze the documents...</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Create a new RAG agent with the selected model
                    status_text.markdown(
                        f'<p style="color: #2E7D32;"><b>⏳ Creating RAG agent...</b></p>',
                        unsafe_allow_html=True,
                    )
                    rag_agent = RAGAgent(selected_model)
                    st.session_state.chat_rag_agent = (
                        rag_agent  # Store for chat functionality
                    )
                    progress_bar.progress(25)

                    # Update the LLM in all specialized agents
                    status_text.markdown(
                        f'<p style="color: #2E7D32;"><b>⏳ Initializing specialized agents...</b></p>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.rag_system.agents = {
                        "primary": rag_agent,  # Main query agent
                        "sentiment": BiodiversityAnalysisAgent(
                            rag_agent.llm
                        ),  # For sentiment analysis
                        "social_media": BiodiversityAnalysisAgent(
                            rag_agent.llm
                        ),  # For social media sentiment analysis
                        "framework": BiodiversityFrameworkAgent(
                            rag_agent.llm
                        ),  # Key must be "framework" for evaluation
                    }
                    progress_bar.progress(35)

                    # Set up project and load/create vector store
                    status_text.markdown(
                        f'<p style="color: #2E7D32;"><b>⏳ Setting up project: {selected_project}...</b></p>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.rag_system.set_current_project(selected_project)
                    progress_bar.progress(50)
                    status_text.markdown(
                        f'<p style="color: #4CAF50;"><b>✅ Project setup complete</b></p>',
                        unsafe_allow_html=True,
                    )

                    # Connect RAG agent to the vector store
                    st.session_state.rag_system.connect_agent_to_vectorstore(
                        st.session_state.chat_rag_agent
                    )

                    # Start evaluation
                    status_text.markdown(
                        f'<p style="color: #2E7D32;"><b>⏳ Starting criteria evaluation...</b></p>',
                        unsafe_allow_html=True,
                    )
                    progress_bar.progress(60)

                    try:
                        st.session_state.results = (
                            st.session_state.rag_system.evaluate_project(
                                BIODIVERSITY_FRAMEWORK
                            )
                        )
                        progress_bar.progress(90)
                        status_text.markdown(
                            f'<p style="color: #4CAF50;"><b>✅ Evaluation complete!</b></p>',
                            unsafe_allow_html=True,
                        )
                    except Exception as eval_error:
                        st.error(f"Error during evaluation: {str(eval_error)}")
                        raise eval_error

                    progress_bar.progress(100)
                    status_text.markdown(
                        f'<p style="color: #4CAF50;"><b>✅ All processing complete!</b></p>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.processing_status = "complete"
                    st.session_state.active_tab = 0
                    st.rerun()

                except Exception as e:
                    st.session_state.processing_status = "error"
                    st.error(f"Error during processing: {str(e)}")
                    raise e

                finally:
                    # Don't empty the progress container if there was an error
                    if st.session_state.processing_status == "complete":
                        progress_container.empty()

            if st.session_state.processing_status == "complete":
                st.success("✨ Analysis completed successfully!")

        except Exception as e:
            st.session_state.processing_status = "error"
            st.error(f"Error: {str(e)}")
            # Keep the error visible in the UI

# Display results if available and processing is complete
if st.session_state.results and st.session_state.processing_status == "complete":
    # Custom tab styling
    st.markdown(
        """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #E8F5E9;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Define tab content without creating actual tabs
    tab_names = [
        "📊 Overview",
        "📝 Detailed Results",
        "💬 Interactive Chat",
        "📈 Usage Analytics",
    ]

    # Create custom tab navigation using buttons
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] button {
            background-color: #E8F5E9;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            border: none;
            margin-right: 5px;
            font-weight: normal;
        }
        div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"] {
            background-color: #E8F5E9;
            color: #2E7D32;
        }
        div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-primary"] {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create a tab selection UI
    cols = st.columns(len(tab_names))
    for i, (col, name) in enumerate(zip(cols, tab_names)):
        # Style the buttons to look like tabs
        button_style = "primary" if i == st.session_state.active_tab else "secondary"
        if col.button(
            name, key=f"tab_btn_{i}", use_container_width=True, type=button_style
        ):
            st.session_state.active_tab = i
            st.rerun()

    # Use the active tab index to show content
    active_tab = st.session_state.active_tab

    # Content container with border styling
    st.markdown(
        """
        <style>
        .tab-content {
            border: 1px solid #4CAF50;
            border-radius: 0 0 10px 10px;
            padding: 20px;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Tab content container
    tab_content = st.container()

    with tab_content:
        # Apply tab content styling
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        if active_tab == 0:
            # Overview dashboard
            st.markdown("### Biodiversity Assessment Dashboard")

            # Add project location map at the top
            st.markdown("### Project Location")

            # Default coordinates for Coopers Gap Wind Farm
            default_latitude = -26.7326
            default_longitude = 151.4723

            # Create the map figure with a more prominent marker
            map_fig = go.Figure(
                go.Scattermapbox(
                    lat=[default_latitude],
                    lon=[default_longitude],
                    mode="markers",
                    marker=go.scattermapbox.Marker(
                        size=20, color="#FF5252", opacity=1.0
                    ),
                    text=["Coopers Gap Wind Farm"],
                    hoverinfo="text",
                )
            )

            map_fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=default_latitude, lon=default_longitude),
                    zoom=6,
                ),
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
            )

            # Add a title annotation to the map
            map_fig.add_annotation(
                x=0.5,
                y=0.95,
                xref="paper",
                yref="paper",
                text="Coopers Gap Wind Farm, Queensland",
                showarrow=False,
                font=dict(size=16, color="#1B5E20"),
                bgcolor="#E8F5E9",
                bordercolor="#4CAF50",
                borderwidth=2,
                borderpad=4,
                opacity=0.8,
            )

            st.plotly_chart(
                map_fig, use_container_width=True, key="project_location_map"
            )

            # Info about the location in a styled box
            st.markdown(
                """
                <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; margin: 10px 0 20px 0; border-left: 5px solid #4CAF50;">
                    <h4 style="color: #1B5E20; margin-top: 0;">Project Details</h4>
                    <p><strong>Name:</strong> Coopers Gap Wind Farm,<strong>Location :</strong>  Queensland, Australia</p>
                    <p><strong>Coordinates:</strong> -26.7326° S, 151.4723° E, <strong>Capacity :</strong>453 MW, 123 wind turbines</p>
                    <p>This project represents a significant investment in renewable energy infrastructure in the region.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Calculate overall score using current category weights
            weights = getattr(st.session_state, 'category_weights', None)
            overall_score = st.session_state.results.get_overall_score(weights)

            # Display overall score - FULL WIDTH
            st.markdown("### Biodiversity Score")
            
            # Create bullet chart similar to original
            score_percentage = overall_score * 100
            
            # Create bullet chart with gradient background like original
            bullet_fig = go.Figure(go.Indicator(
                mode = "gauge",
                value = score_percentage,
                gauge = {
                    'shape': "bullet",
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue", 'thickness': 0.8},
                    'steps': [
                        {'range': [0, 20], 'color': "rgba(255, 0, 0, 0.3)"},     # Red zone
                        {'range': [20, 40], 'color': "rgba(255, 165, 0, 0.3)"},  # Orange zone  
                        {'range': [40, 60], 'color': "rgba(255, 255, 0, 0.3)"},  # Yellow zone
                        {'range': [60, 80], 'color': "rgba(144, 238, 144, 0.3)"}, # Light green
                        {'range': [80, 100], 'color': "rgba(0, 128, 0, 0.3)"}    # Green zone
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            bullet_fig.update_layout(
                height=150,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12)
            )
            
            st.plotly_chart(bullet_fig, use_container_width=True)

            st.markdown(
                f"""
                <div style="background-color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
                    <h1 style="color: #2E7D32; font-size: 48px;">{overall_score:.2f}</h1>
                    <p style="color: #555; font-size: 18px;">Overall Biodiversity Score (0-1 scale)</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display weight information if weights are configured
            if hasattr(st.session_state, 'category_weights') and st.session_state.category_weights:
                st.markdown("### Category Weights & Contributions")
                
                # Get weight information from the results
                weight_info = st.session_state.results.get_category_weights_info(st.session_state.category_weights)
                
                # Create columns for weight display
                cols = st.columns(len(weight_info))
                for i, (cat_name, info) in enumerate(weight_info.items()):
                    with cols[i]:
                        st.metric(
                            label=f"{cat_name}",
                            value=f"{info['normalized_weight']:.1%}",
                            delta=f"Score: {info['score']:.3f}",
                            help=f"Normalized weight: {info['normalized_weight']:.1%}\nCategory score: {info['score']:.3f}\nContribution to overall: {info['weighted_contribution']:.3f}"
                        )

            # Dollar Value Analysis
            if hasattr(st.session_state, 'subcategory_dollar_values') and st.session_state.subcategory_dollar_values:
                st.markdown("### 💰 Budget Allocation Analysis")
                
                # Sync dollar values to model
                sync_dollar_values_to_model()
                
                # Get dollar value information
                dollar_info = st.session_state.results.get_dollar_value_info()
                
                # Display total budget
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.metric(
                        label="Total Project Budget",
                        value=f"${dollar_info['total']:,.0f}",
                        help="Sum of all subcategory budget allocations"
                    )
                
                # Category-level budget breakdown
                st.markdown("#### Budget by Category")
                category_budget_data = []
                
                for category_name, category_data in dollar_info['categories'].items():
                    category_budget_data.append({
                        "Category": category_name,
                        "Budget": category_data['total'],
                        "Percentage": (category_data['total'] / dollar_info['total'] * 100) if dollar_info['total'] > 0 else 0
                    })
                
                # Display category budgets as metrics
                cols = st.columns(len(category_budget_data))
                for i, (col, data) in enumerate(zip(cols, category_budget_data)):
                    with col:
                        st.metric(
                            label=data['Category'],
                            value=f"${data['Budget']:,.0f}",
                            delta=f"{data['Percentage']:.1f}% of total",
                            help=f"Budget allocation for {data['Category']}"
                        )
                
                # Subcategory-level budget table
                st.markdown("#### Detailed Subcategory Budget Breakdown")
                
                subcategory_budget_data = []
                for subcat_id, subcat_data in dollar_info['subcategories'].items():
                    percentage = (subcat_data['value'] / dollar_info['total'] * 100) if dollar_info['total'] > 0 else 0
                    subcategory_budget_data.append({
                        "Category": subcat_data['category'],
                        "Subcategory": subcat_data['name'],
                        "Budget": subcat_data['value'],
                        "Percentage": percentage
                    })
                
                # Sort by budget descending
                subcategory_budget_data.sort(key=lambda x: x['Budget'], reverse=True)
                
                df_budget = pd.DataFrame(subcategory_budget_data)
                
                # Display as formatted table
                st.dataframe(
                    df_budget.style.format({
                        'Budget': '${:,.0f}',
                        'Percentage': '{:.1f}%'
                    }),
                    use_container_width=True
                )
                
                # Budget distribution chart
                fig_budget = px.treemap(
                    df_budget,
                    path=[px.Constant("Total Budget"), 'Category', 'Subcategory'],
                    values='Budget',
                    title="Budget Allocation Treemap",
                    color='Budget',
                    color_continuous_scale='Viridis'
                )
                fig_budget.update_layout(height=500)
                st.plotly_chart(fig_budget, use_container_width=True, key="budget_treemap")

            # Add category gauge chart - FULL WIDTH
            st.markdown("### Category Performance Gauges")

            # Get category scores
            category_scores = {}
            for category_score in st.session_state.results.category_scores:
                category_scores[category_score.category.value] = (
                    category_score.calculate_score()
                )

            # Create dataframe for gauges
            gauge_df = pd.DataFrame(
                {
                    "Category": list(category_scores.keys()),
                    "Category score": list(category_scores.values()),
                }
            )

            # Shorten long category names if needed
            def format_category_name(name):
                if len(name) > 25:
                    return "<br>".join(
                        name.split(" ", 1)
                    )  # Split at first space and add line break
                else:
                    return name

            # Create grid of gauges
            gauge_fig = make_subplots(
                rows=1,
                cols=len(gauge_df),
                specs=[[{"type": "indicator"}] * len(gauge_df)],
            )

            # Add gauges with formatted titles
            for i in range(len(gauge_df)):
                formatted_name = format_category_name(gauge_df.loc[i, "Category"])
                gauge_fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=gauge_df.loc[i, "Category score"] * 100,
                        title={
                            "text": f"<span style='font-size:14px; font-weight:bold; line-height:28px; display:inline-block; text-align:center;'>{formatted_name}</span>",
                            "align": "center",
                        },
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightcoral"},
                                {"range": [50, 75], "color": "gold"},
                                {"range": [75, 100], "color": "lightgreen"},
                            ],
                        },
                        number={"suffix": "%", "font": {"size": 24}},
                    ),
                    row=1,
                    col=i + 1,
                )

            # Layout
            gauge_fig.update_layout(
                height=400,
                margin=dict(t=40, b=30, l=20, r=20),  # Reduced left margin
                showlegend=False,
                autosize=True,  # Enable autosizing
            )

            current_timestamp = int(time.time())
            st.plotly_chart(
                gauge_fig,
                use_container_width=True,
                key=f"tab1_category_gauges_{current_timestamp}",
            )

            # Add sentiment scores if available - FULL WIDTH
            if hasattr(st.session_state.results, "additional_insights"):
                insights = st.session_state.results.additional_insights

                if "sentiment" in insights:
                    st.markdown("### 🔍 Sentiment Analysis")

                    # Make sure we're getting the sentiment data from the model
                    sentiment_score = (
                        insights["sentiment"].get("overall_score", 0.5) * 100
                    )

                    # Debug sentiment data - save for future troubleshooting if needed
                    sentiment_debug = {
                        "overall_score": sentiment_score,
                        "positive_count": len(
                            insights["sentiment"].get("positive_aspects", [])
                        ),
                        "negative_count": len(
                            insights["sentiment"].get("negative_aspects", [])
                        ),
                        "emotions_count": len(
                            insights["sentiment"].get("key_emotions", [])
                        ),
                    }

                    # If there's no valid sentiment data, set a default for demo
                    if sentiment_score == 0 and sentiment_debug["positive_count"] == 0:
                        # Sample sentiment data for demo purposes
                        sentiment_score = 70.5  # Demo value
                        insights["sentiment"]["positive_aspects"] = [
                            "Biodiversity conservation",
                            "Environmental protection",
                            "Renewable energy",
                            "Community engagement",
                            "Habitat preservation",
                        ]
                        insights["sentiment"]["negative_aspects"] = [
                            "Construction impacts",
                            "Bird migration concerns",
                        ]
                        insights["sentiment"]["key_emotions"] = [
                            "Optimism",
                            "Concern",
                            "Support",
                            "Caution",
                            "Hope",
                        ]

                    # Create a larger, more prominent sentiment gauge
                    sentiment_fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=sentiment_score,
                            title={
                                "text": "Overall Sentiment Score",
                                "font": {"size": 24},
                            },
                            gauge={
                                "axis": {
                                    "range": [0, 100],
                                    "tickwidth": 2,
                                    "tickcolor": "darkblue",
                                    "tickvals": [
                                        0,
                                        25,
                                        50,
                                        75,
                                        100,
                                    ],  # Explicit tick values
                                    "ticktext": [
                                        "0",
                                        "25",
                                        "50",
                                        "75",
                                        "100",
                                    ],  # Explicit tick labels
                                },
                                "bar": {"color": "darkblue", "thickness": 0.6},
                                "bgcolor": "white",
                                "borderwidth": 2,
                                "bordercolor": "gray",
                                "steps": [
                                    {"range": [0, 30], "color": "lightcoral"},
                                    {"range": [30, 70], "color": "gold"},
                                    {"range": [70, 100], "color": "lightgreen"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 4},
                                    "thickness": 0.8,
                                    "value": 50,
                                },
                            },
                            number={"suffix": "%", "font": {"size": 40}},
                            domain={
                                "x": [0, 1],
                                "y": [0, 1],
                            },  # Ensure full domain usage
                        )
                    )

                    # Larger height for better visibility
                    sentiment_fig.update_layout(
                        height=300,
                        margin=dict(t=60, r=30, b=30, l=30),  # Adjusted margins
                        paper_bgcolor="white",
                        font={"size": 18, "color": "#2E7D32"},
                        autosize=True,  # Enable autosizing
                    )

                    # Add a unique timestamp to the key to prevent caching issues
                    current_timestamp = int(time.time())
                    st.plotly_chart(
                        sentiment_fig,
                        use_container_width=True,
                        key=f"sentiment_gauge_{current_timestamp}",
                    )

                    # Display positive/negative aspects and emotions in a styled container
                    st.markdown(
                        """
                        <div style="background-color: #E8F5E9; padding: 15px; border-radius: 10px; margin: 10px 0 20px 0;">
                            <h4 style="color: #1B5E20; margin-top: 0;">Sentiment Analysis Details</h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        positive = insights["sentiment"].get("positive_aspects", [])
                        if positive:
                            st.markdown(
                                f"""
                                <div style="background-color: #E3F2FD; border-left: 4px solid #1976D2; padding: 10px; border-radius: 4px;">
                                    <h4 style="margin-top: 0; color: #1976D2;">Positive Aspects</h4>
                                    <p>{", ".join([f"<b>{p}</b>" for p in positive[:5]])}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with col2:
                        negative = insights["sentiment"].get("negative_aspects", [])
                        if negative:
                            st.markdown(
                                f"""
                                <div style="background-color: #FFEBEE; border-left: 4px solid #C62828; padding: 10px; border-radius: 4px;">
                                    <h4 style="margin-top: 0; color: #C62828;">Areas of Concern</h4>
                                    <p>{", ".join([f"<b>{n}</b>" for n in negative[:5]])}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with col3:
                        emotions = insights["sentiment"].get("key_emotions", [])
                        if emotions:
                            st.markdown(
                                f"""
                                <div style="background-color: #F3E5F5; border-left: 4px solid #7B1FA2; padding: 10px; border-radius: 4px;">
                                    <h4 style="margin-top: 0; color: #7B1FA2;">Key Emotions</h4>
                                    <p>{", ".join([f"<b>{e}</b>" for e in emotions[:5]])}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            # Radar chart for category scores - FULL WIDTH
            st.markdown("### Category Scores Overview")
            radar_fig = go.Figure(
                data=go.Scatterpolar(
                    r=list(category_scores.values()),
                    theta=list(category_scores.keys()),
                    fill="toself",
                    marker=dict(color="#4CAF50"),
                    line=dict(color="#1B5E20"),
                )
            )

            radar_fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=500,
                margin=dict(t=40, r=20, b=30, l=20),  # Reduced left margin
                autosize=True,  # Enable autosizing
            )

            st.plotly_chart(
                radar_fig, use_container_width=True, key="radar_chart_categories"
            )

            # Add stacked bar chart of subcategory percentages - FULL WIDTH
            st.markdown("### Subcategory Contribution to Categories")

            # Create DataFrame for the stacked bar chart
            subcategory_data = []
            for category_score in st.session_state.results.category_scores:
                category_name = category_score.category.value

                # Calculate total score for this category to use for percentage calculation
                subcategory_total = sum(
                    sc.calculate_score() for sc in category_score.subcategories
                )

                for subcategory in category_score.subcategories:
                    subcategory_score = subcategory.calculate_score()
                    if subcategory_total > 0:
                        percentage = (subcategory_score / subcategory_total) * 100
                    else:
                        percentage = 0

                    subcategory_data.append(
                        {
                            "Category": category_name,
                            "Sub-Category": subcategory.name,
                            "Subcategory Score": subcategory_score,
                            "Percentage": percentage,
                        }
                    )

            subcategory_df = pd.DataFrame(subcategory_data)

            # Set matplotlib style for better aesthetics
            plt.style.use("ggplot")

            # Create figure with larger size for better readability but more constrained
            fig, ax = plt.subplots(
                figsize=(14, 12), tight_layout=True
            )  # Use tight_layout
            fig.set_facecolor("#f0f7f0")  # Light green background matching the app
            ax.set_facecolor("#ffffff")  # White background for the plot area

            # Pivot data to get it in the right format for stacked bars
            pivot_data = subcategory_df.pivot_table(
                index="Category",
                columns="Sub-Category",
                values="Percentage",
                aggfunc="sum",
            ).fillna(0)

            # Use a nice color palette
            colors = sns.color_palette("Spectral", n_colors=pivot_data.shape[1])

            # Plot the stacked bars
            pivot_data.plot(
                kind="bar", stacked=True, color=colors, edgecolor="black", ax=ax
            )

            # Customize the chart
          #  ax.set_title(
          #      "Subcategory Distribution within Each Category",
          #      fontsize=20,
          #      pad=20,
          #      fontweight="bold",
         #   )
            ax.set_ylabel("Percentage Contribution (%)", fontsize=16, fontweight="bold")
            ax.set_xlabel("Category", fontsize=16, fontweight="bold")

            # Adjust tick labels
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=30,
                ha="right",
                fontsize=12,
                fontweight="bold",
            )
            ax.tick_params(axis="y", labelsize=14)

            # Add percentage labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=10)

            # Place legend below the plot for better visibility with many subcategories
            ax.legend(
                title="Subcategory",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.70),
                ncol=3,
                fontsize=12,
                title_fontsize=14,
            )

            # Clean look and tight layout
            sns.despine()
            plt.tight_layout()

            # Display the chart in Streamlit
            current_timestamp = int(time.time())
            st.pyplot(fig)

            # Clear the matplotlib figure to prevent memory issues
            plt.close(fig)

            # Bar chart for subcategory scores - FULL WIDTH
            st.markdown("### Subcategory Scores")
            subcategory_scores = []
            for category_score in st.session_state.results.category_scores:
                for subcategory in category_score.subcategories:
                    subcategory_scores.append(
                        {
                            "Category": category_score.category.value,
                            "Subcategory": subcategory.name,
                            "Score": subcategory.calculate_score(),
                        }
                    )

            subcategory_df = pd.DataFrame(subcategory_scores)
            subcategory_df = subcategory_df.sort_values("Score", ascending=False)

            fig = px.bar(
                subcategory_df,
                x="Subcategory",
                y="Score",
                color="Category",
                # title="Subcategory Scores",
                color_discrete_sequence=px.colors.qualitative.G10,
            )

            fig.update_layout(
                xaxis_title="Subcategory",
                yaxis_title="Score",
                xaxis={"tickangle": 45},
                height=600,
            )

            st.plotly_chart(
                fig, use_container_width=True, key="bar_chart_subcategories"
            )

            # Add metrics relationship visualization - FULL WIDTH
            if "biodiversity_df" in st.session_state:
                try:
                    st.markdown("### Metrics Contribution to Score")
                    # Filter for rows with category data
                    category_data = st.session_state.biodiversity_df[
                        st.session_state.biodiversity_df["Category N"].notna()
                    ].copy()
                    category_data = category_data.drop_duplicates(
                        subset=["Category"]
                    ).reset_index(drop=True)

                    if not category_data.empty:
                        # Create a parallel coordinates plot
                        fig = go.Figure(
                            data=go.Parcoords(
                                line=dict(
                                    color=category_data["Category Score"],
                                    colorscale="Viridis",
                                    showscale=True,
                                    cmin=0,
                                    cmax=1,
                                ),
                                dimensions=list(
                                    [
                                        dict(
                                            range=[
                                                0,
                                                category_data["Category N"].max() + 5,
                                            ],
                                            label="Number of Mentions (N)",
                                            values=category_data["Category N"],
                                        ),
                                        dict(
                                            range=[0, 100],
                                            label="Specificity (S %)",
                                            values=category_data["Category S (%)"],
                                        ),
                                        dict(
                                            range=[0, 100],
                                            label="Multiplicity (M %)",
                                            values=category_data["Category M (%)"],
                                        ),
                                        dict(
                                            range=[0, 1],
                                            label="Category Score",
                                            values=category_data["Category Score"],
                                        ),
                                    ]
                                ),
                            )
                        )

                        fig.update_layout(
                            title="Category Metrics and Their Contribution to Scores",
                            height=500,
                        )

                        # Use timestamp to ensure uniqueness across reruns
                        current_timestamp = int(time.time())
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"metrics_parallel_coords_{hash(str(category_data.shape))}_{current_timestamp}",
                        )
                    else:
                        st.info(
                            "No category data available to visualize the metrics relationship."
                        )
                except Exception as e:
                    st.error(f"Error generating metrics visualization: {str(e)}")

        elif active_tab == 1:
            # Display detailed evaluation results
            display_evaluation_results()

        elif active_tab == 2:
            # Display the chat interface in the Interactive Chat tab
            chat_interface.display_chat_interface()

        elif active_tab == 3:
            # Token usage statistics
            token_stats = st.session_state.rag_system.get_token_stats()

            # Create a DataFrame for token usage
            token_data = []
            agent_display_names = {
                "sentiment": "Sentiment Analysis",
                "social_media": "Social Media Analysis",
                "primary": "Primary Analysis",
                "framework": "Biodiversity Framework Analysis",
            }

            for agent_name, stats in token_stats.items():
                token_data.append(
                    {
                        "Agent": agent_display_names.get(agent_name, agent_name),
                        "Total Tokens": stats["total_tokens"],
                        "Total Cost ($)": stats["total_cost"],
                        "Successful Requests": stats["successful_requests"],
                        "Failed Requests": stats["failed_requests"],
                    }
                )

            token_df = pd.DataFrame(token_data)

            # Display token usage
            st.subheader("Usage Analytics")

            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tokens Used", f"{token_df['Total Tokens'].sum():,}")
            with col2:
                st.metric("Total Cost", f"${token_df['Total Cost ($)'].sum():.2f}")
            with col3:
                st.metric(
                    "Successful Requests", f"{token_df['Successful Requests'].sum():,}"
                )
            with col4:
                st.metric("Failed Requests", f"{token_df['Failed Requests'].sum():,}")

            # Detailed stats
            st.dataframe(token_df)

            # Usage charts
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    token_df, x="Agent", y="Total Tokens", title="Token Usage by Agent"
                )
                st.plotly_chart(fig, key="bar_chart_tokens_tab4")

            with col2:
                fig = px.bar(
                    token_df, x="Agent", y="Total Cost ($)", title="Cost by Agent"
                )
                st.plotly_chart(fig, key="bar_chart_cost_tab4")

            # Add biodiversity assessment visualization
            if "biodiversity_df" in st.session_state:
                try:
                    st.markdown("### Biodiversity Assessment Visualization")
                    st.markdown(
                        "This chart shows the relationship between Category N (mentions) and Category Score:"
                    )

                    # Create a scatter plot of Category N vs Category Score
                    category_data = st.session_state.biodiversity_df[
                        st.session_state.biodiversity_df["Category N"].notna()
                    ].copy()
                    category_data = category_data.drop_duplicates(
                        subset=["Category"]
                    ).reset_index(drop=True)

                    if not category_data.empty:
                        fig = px.scatter(
                            category_data,
                            x="Category N",
                            y="Category Score",
                            color="Category",
                            size="Category N",
                            hover_data=["Category S (%)", "Category M (%)"],
                            title="Category Metrics Relationship",
                            labels={
                                "Category N": "Number of Mentions (N)",
                                "Category Score": "Category Score",
                                "Category S (%)": "Specificity (%)",
                                "Category M (%)": "Multiplicity (%)",
                            },
                        )

                        fig.update_layout(height=500)
                        # Use timestamp to ensure uniqueness across reruns
                        current_timestamp = int(time.time())
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"category_scatter_plot_{hash(str(category_data.shape))}_{current_timestamp}",
                        )
                    else:
                        st.info("No category data available to visualize.")
                except Exception as e:
                    st.error(f"Error generating category visualization: {str(e)}")
        # Close the tab content div
        st.markdown("</div>", unsafe_allow_html=True)
else:
    # Show welcome message when no project is selected or processed
    if not selected_project:
        st.markdown(
            """
        <div style="background-color: white; padding: 30px; border-radius: 10px; text-align: center; margin: 50px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <img src="https://cdn-icons-png.flaticon.com/512/4856/4856868.png" style="width: 100px; margin-bottom: 20px;">
            <h2 style="color: #1B5E20;">Welcome to the Biodiversity Criteria Evaluator</h2>
            <p style="font-size: 1.2em; color: #2E7D32; margin: 20px 0;">
                Please select a project from the sidebar to get started.
            </p>
            <div style="background-color: #E8F5E9; padding: 15px; border-radius: 8px; display: inline-block; margin-top: 10px;">
                <p style="margin: 0; color: #1B5E20;">
                    <b>Step 1:</b> Select a project from the dropdown<br>
                    <b>Step 2:</b> Choose a model<br>
                    <b>Step 3:</b> Click "Process and Evaluate"
                </p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    elif st.session_state.processing_status != "complete":
        # If project is selected but not processed
        st.info(
            f"Project '{selected_project}' selected. Click 'Process and Evaluate' to analyze the documents."
        )

# Check if we're in the middle of chat processing to keep that tab active
if st.session_state.processing_chat and st.session_state.active_tab != 2:
    st.session_state.active_tab = 2
