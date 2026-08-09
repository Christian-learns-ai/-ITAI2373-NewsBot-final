import plotly.express as px
import pandas as pd

def plot_topic_evolution(timeline_df: pd.DataFrame):
    """Plots interactive Plotly topic proportions over time."""
    if timeline_df is not None and not timeline_df.empty:
        fig = px.line(
            timeline_df,
            x=timeline_df.index,
            y=timeline_df.columns,
            title="📈 Topic Evolution Over Time",
            labels={"value": "Topic Proportion", "date": "Date"},
        )
        return fig
    return None
