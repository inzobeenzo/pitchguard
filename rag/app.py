import streamlit as st
from engine import build_index, explain

@st.cache_resource
def get_index():
    return build_index()

coll = get_index()
st.title("PitchGuard — injury-risk explainer")
text = st.text_area("Top risk factors (one per line)", "sharp rise in acute-to-chronic workload ratio\ndeclining fastball velocity\nlowered release point")

if st.button("Explain"):
    with st.spinner("Retrieving evidence and generating explanation..."):
        r = explain(coll, [f for f in text.splitlines() if f.strip()])
    st.write(r["answer"])
    st.caption("Grounded in: " + ", ".join(r["sources"]))