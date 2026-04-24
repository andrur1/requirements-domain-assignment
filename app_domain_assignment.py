import streamlit as st
import duckdb
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="AI Domain Assignment Dashboard",
    layout="wide"
)

st.title("🤖 AI Domain Assignment Dashboard")
st.caption("AI-assisted responsible domain recommendation for requirements")

con = duckdb.connect("requirements.duckdb")

df = con.execute("""
    SELECT *
    FROM domain_assignment_output
    ORDER BY requirement_id
""").fetchdf()

df = df.fillna("")

# KPIs
total_requirements = len(df)
ambiguous_cases = len(df[df["ambiguous"] == True])
high_confidence = len(df[df["confidence"] == "High"])

c1, c2, c3 = st.columns(3)
c1.metric("Total Requirements", total_requirements)
c2.metric("Ambiguous Cases", ambiguous_cases)
c3.metric("High Confidence", high_confidence)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("Suggested Domain Distribution")

    domain_counts = (
        df["suggested_domain"]
        .value_counts()
        .reset_index()
    )
    domain_counts.columns = ["suggested_domain", "count"]

    domain_chart = alt.Chart(domain_counts).mark_bar().encode(
        x=alt.X("count:Q", title="Count"),
        y=alt.Y("suggested_domain:N", title="Domain", sort="-x"),
        color=alt.Color("suggested_domain:N", legend=None)
    ).properties(height=300)

    st.altair_chart(domain_chart, use_container_width=True)

with right:
    st.subheader("Confidence Distribution")

    confidence_order = ["Low", "Medium", "High"]

    confidence_counts = (
        df["confidence"]
        .value_counts()
        .reindex(confidence_order, fill_value=0)
        .reset_index()
    )
    confidence_counts.columns = ["confidence", "count"]

    confidence_chart = alt.Chart(confidence_counts).mark_bar(size=45).encode(
        x=alt.X(
            "confidence:N",
            title="Confidence",
            sort=confidence_order,
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y("count:Q", title="Count"),
        color=alt.Color(
            "confidence:N",
            scale=alt.Scale(
                domain=["Low", "Medium", "High"],
                range=["#ef4444", "#facc15", "#22c55e"]
            ),
            legend=None
        )
    ).properties(height=300)

    text = confidence_chart.mark_text(
        align="center",
        baseline="bottom",
        dy=-5,
        fontSize=14
    ).encode(text="count:Q")

    st.altair_chart(confidence_chart + text, use_container_width=True)

st.divider()

st.subheader("Ambiguous Requirements")

ambiguous_df = df[df["ambiguous"] == True].copy()

if ambiguous_df.empty:
    st.success("No ambiguous requirements detected.")
else:
    st.dataframe(
        ambiguous_df[
            [
                "requirement_id",
                "suggested_domain",
                "secondary_domain",
                "confidence",
                "score_top1",
                "score_top2",
                "rationale"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

st.divider()

st.subheader("Domain Assignment Results")

display_df = df[
    [
        "requirement_id",
        "suggested_domain",
        "secondary_domain",
        "confidence",
        "ambiguous",
        "rationale",
        "score_top1",
        "score_top2"
    ]
].copy()

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

st.divider()

st.subheader("Export")

st.download_button(
    label="Download domain assignment CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="domain_assignment_output.csv",
    mime="text/csv"
)
