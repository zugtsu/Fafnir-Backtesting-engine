"""
Fafnir backtest GUI.

    streamlit run gui/app.py
"""

import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
# Pages import `gui.common` and `saty_bt`, both rooted at the repo.
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

st.set_page_config(page_title="Fafnir Backtester", page_icon="📈", layout="wide")

from gui.style import apply_glass  # noqa: E402

apply_glass()

pages = st.navigation([
    st.Page(HERE / "views" / "backtest.py", title="Backtest", icon="▶️", default=True),
    st.Page(HERE / "views" / "results.py", title="Results", icon="📊"),
    st.Page(HERE / "views" / "data.py", title="Data", icon="🗂️"),
    st.Page(HERE / "views" / "indicators.py", title="Indicators", icon="🧪"),
])
pages.run()

st.sidebar.caption("Research tool, not financial advice. Past results don't predict future ones.")
