import json
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from search.hybrid import HybridSearchEngine, SearchResult, hybrid_search


load_dotenv()


st.set_page_config(
    page_title="Enterprise Policy Hybrid Search Assistant",
    layout="centered",
)


@st.cache_resource
def load_engine():
    """Load policies and embeddings once per session."""
    data_path = Path(__file__).parent / "data" / "policies.json"
    return HybridSearchEngine(data_path)


def main() -> None:
    st.title("Enterprise Policy Hybrid Search Assistant")
    st.info(
        "Hybrid search combines lexical precision (keyword matching) and semantic similarity (vector embeddings). "
        "Adjust the slider to see how ranking behavior changes."
    )

    with st.spinner("Loading embedding model..."):
        engine = load_engine()
    st.success("Embedding model loaded successfully.")

    policies = engine._documents
    regions = sorted({p.get("region") for p in policies if p.get("region")})
    region_options = ["All"] + regions
    selected_region = st.selectbox("Filter by Region", region_options, index=0)
    if selected_region == "All":
        filtered_policies = policies
    else:
        filtered_policies = [p for p in policies if p.get("region") == selected_region]

    search_mode = st.radio(
        "Search Mode",
        options=["Hybrid", "Keyword Only", "Semantic Only"],
        index=0,
    )
    if search_mode == "Keyword Only":
        keyword_weight = st.slider(
            "Keyword Weight",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            disabled=True,
        )
    elif search_mode == "Semantic Only":
        keyword_weight = st.slider(
            "Keyword Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            disabled=True,
        )
    else:
        keyword_weight = st.slider(
            "Keyword Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            disabled=False,
        )

    vector_weight = 1.0 - keyword_weight
    st.slider(
        "Vector Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(vector_weight),
        step=0.05,
        disabled=True,
    )

    st.divider()
    st.subheader("Ask a Policy Question")
    query = st.text_input("Query", placeholder="e.g. international data movement approval", key="query")
    st.caption("Example: international data movement approval")
    st.divider()

    search_clicked = st.button("Search")

    if not search_clicked:
        st.caption("Enter a query and click Search to run hybrid search.")
        return

    if not query or not query.strip():
        st.warning("Please enter a query.")
        return

    results = hybrid_search(
        query=query.strip(),
        policies=filtered_policies,
        keyword_weight=keyword_weight,
        vector_weight=vector_weight,
    )

    if not results:
        st.info("No matching policies found.")
        return

    st.divider()
    st.subheader("Hybrid Score Breakdown")
    st.markdown("### Results")

    # Stacked bar chart for top 5 search results
    top_results = results[:5]
    titles = [r.title for r in top_results]
    keyword_contrib = [keyword_weight * r.keyword_score for r in top_results]
    vector_contrib = [vector_weight * r.vector_score for r in top_results]

    x = range(len(top_results))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, keyword_contrib, label="Keyword contribution")
    ax.bar(x, vector_contrib, bottom=keyword_contrib, label="Semantic contribution")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(titles, rotation=20, ha="right")
    ax.set_title("Top 5 Hybrid Search Results (Stacked View)")
    ax.set_ylabel("Score contribution")
    ax.legend()

    st.pyplot(fig)

    st.divider()
    st.subheader("Detailed Policy Matches")
    st.caption("Below are the full ranked results with score breakdown.")

    for result in results:
        content_preview = result.content[:300] + ("..." if len(result.content) > 300 else "")

        st.markdown(f"**{result.title}**")
        st.caption(result.owner)

        st.markdown("**Hybrid Score Breakdown:**")
        st.markdown(
            f"- Keyword Score: {result.keyword_score:.3f}\n"
            f"- Vector Score: {result.vector_score:.3f}\n"
            f"- Final Score: {result.final_score:.3f}"
        )
        st.write(result.content)
        st.divider()


if __name__ == "__main__":
    main()
