"""Interactive Streamlit dashboard backed by the trained local artifacts."""
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from api.service import RecommendationService

st.set_page_config(page_title="Intelligent Recommendation Engine", page_icon="🎬", layout="wide")
st.title("🎬 Intelligent Recommendation Engine")
st.caption("Two-tower retrieval, FAISS candidate generation, and neural re-ranking — trained locally on MovieLens 100K.")

@st.cache_resource
def load_service(): return RecommendationService()

try:
    service = load_service()
except RuntimeError as error:
    st.error(str(error)); st.stop()

users = sorted(service.metadata["user_map"].keys(), key=int)
left, right = st.columns([1, 3])
with left:
    user = st.selectbox("MovieLens user", users)
    k = st.slider("Recommendations", 3, 20, 10)
    st.subheader("Model")
    st.json({key: service.model_info()[key] for key in ("dataset", "embedding_dim", "retrieval_backend")})
with right:
    recommendations = service.recommend(user, k)
    st.subheader(f"Personalized picks for user {user}")
    st.dataframe(pd.DataFrame(recommendations), use_container_width=True, hide_index=True)
    chart = pd.DataFrame(recommendations).set_index("title")[["score", "retrieval_score"]]
    st.bar_chart(chart)
st.divider()
item = st.selectbox("Explore similar films", service.items.item_raw.astype(str).tolist(), format_func=lambda x: service.items.iloc[service.metadata["item_map"][x]].title)
st.dataframe(pd.DataFrame(service.similar_items(item, k)), use_container_width=True, hide_index=True)
