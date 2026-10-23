"""Frosted-glass look for the GUI: a glowing background with translucent, blurred panels on top."""

import streamlit as st

GLASS_CSS = """
<style>
:root {
  --glass: rgba(255, 255, 255, 0.07);
  --glass-strong: rgba(255, 255, 255, 0.11);
  --glass-edge: rgba(255, 255, 255, 0.16);
  --glass-shine: rgba(255, 255, 255, 0.22);
  --blur: blur(22px) saturate(170%);
  --radius: 20px;
  --accent: #8B7CF6;
}

/* Glowing background: the blur needs color behind it to read as glass. */
.stApp {
  background:
    radial-gradient(900px 600px at 8% 0%, rgba(124, 92, 255, 0.45), transparent 60%),
    radial-gradient(800px 600px at 100% 20%, rgba(34, 211, 238, 0.28), transparent 60%),
    radial-gradient(900px 700px at 60% 110%, rgba(236, 72, 153, 0.25), transparent 60%),
    radial-gradient(700px 500px at 30% 60%, rgba(59, 130, 246, 0.18), transparent 60%),
    #0b0d1a;
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }

/* Sidebar: one tall pane of glass. */
[data-testid="stSidebar"] {
  background: rgba(16, 18, 36, 0.45);
  -webkit-backdrop-filter: var(--blur);
  backdrop-filter: var(--blur);
  border-right: 1px solid var(--glass-edge);
}
[data-testid="stSidebar"] > div { background: transparent; }

/* Panels: stat cards, expanders, bordered boxes, charts, tables, alerts. */
[data-testid="stMetric"],
[data-testid="stExpander"] details,
[class*="st-key-glass"],
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stFileUploaderDropzone"],
[data-testid="stCode"] {
  background: var(--glass);
  -webkit-backdrop-filter: var(--blur);
  backdrop-filter: var(--blur);
  border: 1px solid var(--glass-edge);
  border-radius: var(--radius);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25), inset 0 1px 0 var(--glass-shine);
}
[data-testid="stMetric"] { padding: 14px 18px; }
[data-testid="stPlotlyChart"] { padding: 8px; overflow: hidden; }
[data-testid="stDataFrame"] { padding: 4px; overflow: hidden; }
[data-testid="stExpander"] details { overflow: hidden; }
[data-testid="stExpander"] summary:hover { background: var(--glass); }
[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.5px; }

[data-testid="stAlert"] > div {
  background: var(--glass-strong);
  -webkit-backdrop-filter: var(--blur);
  backdrop-filter: var(--blur);
  border: 1px solid var(--glass-edge);
  border-radius: var(--radius);
}

/* Inputs: frosted pills. */
[data-baseweb="select"] > div,
[data-testid="stNumberInputContainer"],
[data-testid="stTextInputRootElement"],
[data-testid="stTextAreaRootElement"],
[data-testid="stDateInput"] [data-baseweb="input"] {
  background: rgba(255, 255, 255, 0.06) !important;
  border: 1px solid var(--glass-edge) !important;
  border-radius: 14px !important;
}
[data-testid="stNumberInputContainer"] > div,
[data-testid="stTextInputRootElement"] > div,
[data-testid="stTextAreaRootElement"] > div,
[data-baseweb="base-input"], .stTextArea textarea { background: transparent !important; }
[data-testid="stNumberInput"] button { background: transparent; border: none; }
.stTextArea textarea { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }

/* Buttons: glass pills; the primary one glows in the accent color. */
.stButton > button, .stDownloadButton > button, [data-testid="stBaseButton-secondary"] {
  background: var(--glass-strong);
  -webkit-backdrop-filter: var(--blur);
  backdrop-filter: var(--blur);
  border: 1px solid var(--glass-edge);
  border-radius: 999px;
  box-shadow: inset 0 1px 0 var(--glass-shine);
  transition: transform .12s ease, background .2s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: rgba(255, 255, 255, 0.16); }
.stButton > button:active { transform: scale(0.97); }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, rgba(139, 124, 246, 0.95), rgba(99, 102, 241, 0.85));
  border: 1px solid rgba(255, 255, 255, 0.28);
  box-shadow: 0 8px 24px rgba(124, 92, 255, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.35);
  color: #fff;
}

.stButton > button:disabled, .stDownloadButton > button:disabled {
  opacity: 0.4;
  box-shadow: none;
  cursor: not-allowed;
}

/* Tabs: a glass segmented control. */
[data-testid="stTabs"] [role="tablist"] {
  background: var(--glass);
  -webkit-backdrop-filter: var(--blur);
  backdrop-filter: var(--blur);
  border: 1px solid var(--glass-edge);
  border-radius: 999px;
  padding: 4px;
  gap: 4px;
  width: fit-content;
}
[data-testid="stTab"] { border-radius: 999px; padding: 6px 16px; height: auto; }
[data-testid="stTab"][aria-selected="true"] { background: rgba(139, 124, 246, 0.35); }
[data-testid="stTab"][aria-selected="true"] p { color: #fff; }
[data-testid="stTab"] { border-bottom: none !important; }
[data-testid="stTab"]::after, [data-testid="stTab"]::before { display: none !important; }
[data-testid="stTabs"] [role="tablist"] > div:not([data-testid="stTab"]),
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }

/* Page navigation in the sidebar. */
[data-testid="stSidebarNav"] a { border-radius: 12px; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: rgba(139, 124, 246, 0.28);
  box-shadow: inset 0 1px 0 var(--glass-shine);
}

h1 { letter-spacing: -1px; }
</style>
"""


def apply_glass() -> None:
    st.markdown(GLASS_CSS, unsafe_allow_html=True)
