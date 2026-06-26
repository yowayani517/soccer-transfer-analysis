import os, sys, html
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocess import load_and_clean, build_features, transform_single, TARGET
from src.model import train, predict_value, predict_all, feature_importance

# ── ページ設定 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Transfer Alpha · Valuation Terminal", page_icon="⚽",
                   layout="wide", initial_sidebar_state="collapsed")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "final_data.csv")

# ── Transfer Alpha Terminal テーマ ──────────────────────────────────────────
ACCENT, POS, NEG = "#d6a23a", "#48c07a", "#e0566c"
PLOT_BG, PAPER = "#0d0d0f", "rgba(0,0,0,0)"
MONO = "IBM Plex Mono, monospace"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans+JP:wght@400;500;600;700&display=swap');
:root{
 --bg:#0a0a0b; --panel:#121214; --panel2:#17171a; --inset:#0d0d0f;
 --line:rgba(255,255,255,0.07); --line2:rgba(255,255,255,0.13);
 --text:#ededf0; --muted:#8a8b91; --faint:#56575d;
 --accent:#d6a23a; --accent-dim:rgba(214,162,58,0.13);
 --pos:#48c07a; --pos-dim:rgba(72,192,122,0.12);
 --neg:#e0566c; --neg-dim:rgba(224,86,108,0.12);
}
.stApp{background:#0a0a0b;background-image:radial-gradient(1000px 520px at 82% -12%, rgba(214,162,58,0.06), transparent 62%);color:var(--text);font-family:'IBM Plex Sans JP',system-ui,sans-serif;}
header[data-testid="stHeader"]{display:none;}
[data-testid="stSidebar"]{display:none;}
.block-container{max-width:1340px;padding:14px 26px 60px;}
.mono{font-family:'IBM Plex Mono',monospace;}
::selection{background:rgba(214,162,58,0.3);}
@keyframes marquee{from{transform:translateX(0)}to{transform:translateX(-50%)}}
@keyframes pulseDot{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes barGrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
/* タブを端末風ナビに */
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{flex:1;justify-content:center;font-family:'IBM Plex Sans JP',sans-serif;font-weight:600;font-size:13.5px;color:var(--muted);padding:12px 8px;}
.stTabs [data-baseweb="tab"]:hover{color:var(--text);}
.stTabs [aria-selected="true"]{color:var(--text)!important;}
.stTabs [data-baseweb="tab-highlight"]{background:var(--accent)!important;height:2px!important;box-shadow:0 0 10px var(--accent-dim);}
.stTabs [data-baseweb="tab-border"]{background:transparent;}
/* 入力系をダークに */
div[data-baseweb="select"]>div{background:var(--inset)!important;border:1px solid var(--line2)!important;color:var(--text)!important;border-radius:8px!important;}
.stSelectbox label,.stSlider label{color:var(--muted)!important;font-size:12px!important;}
.stSlider [data-baseweb="slider"]{padding-top:4px;}
div[data-testid="stSlider"] [role="slider"]{background:var(--accent)!important;}
.stRadio [role="radiogroup"]{gap:4px;}
hr{border-color:var(--line);}
</style>
""", unsafe_allow_html=True)

# ── ユーティリティ ──────────────────────────────────────────────────────────
def fmt_eur(v):
    if v >= 1e6: return f"€{v/1e6:.1f}M"
    if v >= 1e3: return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

def fmt_jpy(v):
    m = v / 1e6 * 160
    return f"約{m/10000:.0f}億円" if m >= 10000 else f"約{m:.0f}百万円"

def esc(s): return html.escape(str(s))

PLOT_FONT = dict(family=MONO, color="#cfd0d4", size=11)
def style_fig(fig, h):
    fig.update_layout(height=h, paper_bgcolor=PAPER, plot_bgcolor=PLOT_BG,
                      font=PLOT_FONT, margin=dict(l=0, r=0, t=8, b=0), showlegend=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig

# ── データ・モデル ──────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_clean(DATA_PATH)

@st.cache_resource
def get_model(_df):
    X, y, encoders, names = build_features(_df)
    model, metrics, splits = train(X, y)
    return model, encoders, names, metrics, splits, predict_all(model, X)

if not os.path.exists(DATA_PATH):
    st.error("`data/final_data.csv` が見つかりません。"); st.stop()

df = get_data()
with st.spinner("⚙️ XGBoostモデルを学習中… 初回のみ30〜60秒"):
    model, encoders, feature_names, metrics, splits, all_preds = get_model(df)
df["predicted"] = all_preds
df["gap"] = df["predicted"] - df[TARGET]
df["gap_ratio"] = df["gap"] / df[TARGET]

avg_v, max_v = df[TARGET].mean(), df[TARGET].max()
max_name = df.loc[df[TARGET].idxmax(), "name"]

# ── ヘッダーバー ────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;height:56px;padding:0 4px;border-bottom:1px solid var(--line);margin-bottom:0;">
  <div style="display:flex;align-items:center;gap:13px;">
    <div class="mono" style="width:30px;height:30px;border:1px solid var(--accent);border-radius:6px;display:flex;align-items:center;justify-content:center;color:var(--accent);font-weight:700;font-size:16px;box-shadow:0 0 18px var(--accent-dim);">α</div>
    <div style="display:flex;flex-direction:column;line-height:1.1;">
      <span class="mono" style="font-weight:700;font-size:14px;letter-spacing:.16em;">TRANSFER&nbsp;ALPHA</span>
      <span class="mono" style="font-size:9.5px;letter-spacing:.26em;color:var(--faint);text-transform:uppercase;">Player Valuation Terminal</span>
    </div>
  </div>
  <div class="mono" style="display:flex;align-items:center;gap:9px;font-size:11px;">
    <span style="display:inline-flex;align-items:center;gap:7px;padding:5px 11px;border:1px solid var(--line);border-radius:6px;color:var(--muted);"><span style="width:7px;height:7px;border-radius:50%;background:var(--pos);box-shadow:0 0 8px var(--pos);animation:pulseDot 2s ease-in-out infinite;"></span><span style="color:var(--pos);">LIVE</span></span>
    <span style="padding:5px 11px;border:1px solid var(--line);border-radius:6px;color:var(--muted);">MODEL <span style="color:var(--text);">XGB · v2.1</span></span>
    <span style="padding:5px 11px;border:1px solid var(--line);border-radius:6px;color:var(--muted);">CV R² <span style="color:var(--accent);">{metrics['cv_r2_mean']:.3f}</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ティッカー（実データ） ──────────────────────────────────────────────────
tick = df.nlargest(22, TARGET)
tick_html = ""
for _, r in tick.iterrows():
    col = POS if r["gap_ratio"] >= 0 else NEG
    ar = "▲" if r["gap_ratio"] > 0 else "▼"
    tick_html += f'<span style="display:inline-flex;align-items:baseline;gap:8px;padding:0 18px;">'\
                 f'<span style="color:var(--text);font-weight:500;">{esc(str(r["name"]).upper())}</span>'\
                 f'<span style="color:var(--muted);">{fmt_eur(r[TARGET])}</span>'\
                 f'<span style="color:{col};font-size:10px;">{ar}</span></span>'
st.markdown(f"""
<div style="border-bottom:1px solid var(--line);background:var(--inset);overflow:hidden;height:34px;display:flex;align-items:center;position:relative;margin-bottom:22px;">
  <div class="mono" style="position:absolute;left:0;z-index:2;height:34px;display:flex;align-items:center;padding:0 12px;background:var(--inset);border-right:1px solid var(--line);font-size:10px;letter-spacing:.18em;color:var(--accent);">VALUE FEED</div>
  <div class="mono" style="display:flex;white-space:nowrap;animation:marquee 52s linear infinite;padding-left:120px;font-size:12px;"><div style="display:flex;">{tick_html}</div><div style="display:flex;">{tick_html}</div></div>
</div>
""", unsafe_allow_html=True)

# ── タイトル ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:14px;margin-bottom:16px;">
  <div>
    <div class="mono" style="font-size:11px;letter-spacing:.22em;color:var(--accent);margin-bottom:7px;">MARKET&nbsp;INTELLIGENCE</div>
    <div style="font-size:26px;font-weight:700;letter-spacing:-.01em;">移籍市場バリュエーション</div>
    <div style="margin-top:7px;color:var(--muted);font-size:13.5px;max-width:640px;line-height:1.6;">{len(df):,}名の選手データを学習したXGBoostモデルが市場価値を推定し、<span style="color:var(--text);">市場が見落とした割安な選手</span>を自動で発掘する。</div>
  </div>
  <div class="mono" style="display:flex;gap:8px;font-size:10.5px;">
    <span style="padding:5px 10px;border:1px solid var(--line);border-radius:5px;color:var(--muted);">SOURCE <span style="color:var(--text);">Transfermarkt</span></span>
    <span style="padding:5px 10px;border:1px solid var(--line);border-radius:5px;color:var(--muted);">N <span style="color:var(--text);">{len(df):,}</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIストリップ ───────────────────────────────────────────────────────────
kpis = [
    ("分析対象", f"{len(df):,}", "登録選手数", "var(--text)"),
    ("平均市場価値", fmt_eur(avg_v), fmt_jpy(avg_v), "var(--text)"),
    ("最高市場価値", fmt_eur(max_v), esc(max_name), "var(--text)"),
    ("モデル精度 R²", f"{metrics['cv_r2_mean']:.3f}", f"±{metrics['cv_r2_std']:.3f} · 5-fold CV", "var(--accent)"),
    ("予測誤差 MAE", fmt_eur(metrics["test_mae_eur"]), "テスト平均誤差", "var(--text)"),
]
cells = ""
for i, (lab, val, sub, c) in enumerate(kpis):
    br = "" if i == 4 else "border-right:1px solid var(--line);"
    cells += f'<div style="padding:18px 20px;{br}"><div class="mono" style="font-size:10px;letter-spacing:.16em;color:var(--faint);text-transform:uppercase;margin-bottom:9px;">{lab}</div>'\
             f'<div class="mono" style="font-size:25px;font-weight:600;color:{c};">{val}</div>'\
             f'<div style="font-size:11px;color:var(--muted);margin-top:5px;">{sub}</div></div>'
st.markdown(f'<div style="display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);border-radius:12px;background:var(--panel);overflow:hidden;margin-bottom:24px;">{cells}</div>', unsafe_allow_html=True)

# ── タブ ────────────────────────────────────────────────────────────────────
t_bar, t_pred, t_ana, t_mod = st.tabs(["01　コスパ発掘", "02　価値を予測", "03　データ分析", "04　モデル詳細"])

def panel_open(): return '<div style="border:1px solid var(--line);border-radius:12px;background:var(--panel);'
def table_rows(sub, accent_pos):
    rows = ""
    for i, (_, r) in enumerate(sub.iterrows()):
        gr = r["gap_ratio"]
        col = POS if gr >= 0 else NEG
        barw = min(100, abs(gr * 100))
        gap = f'{"+" if gr>=0 else ""}{gr*100:.0f}%'
        rows += f'<div style="display:grid;grid-template-columns:24px 1fr 42px 30px 62px 62px 92px;gap:10px;align-items:center;padding:9px 16px;border-top:1px solid var(--line);">'\
            f'<span class="mono" style="font-size:11px;color:var(--faint);">{i+1:02d}</span>'\
            f'<div style="min-width:0;"><div style="font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{esc(r["name"])}</div>'\
            f'<div class="mono" style="font-size:10px;color:var(--faint);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{esc(r["team"])}</div></div>'\
            f'<span class="mono" style="font-size:10px;color:var(--muted);border:1px solid var(--line2);border-radius:4px;padding:2px 0;text-align:center;">{esc(r["position"])}</span>'\
            f'<span class="mono" style="font-size:12px;color:var(--muted);">{int(r["age"])}</span>'\
            f'<span class="mono" style="font-size:12px;color:var(--muted);text-align:right;">{fmt_eur(r[TARGET])}</span>'\
            f'<span class="mono" style="font-size:12px;color:var(--text);text-align:right;">{fmt_eur(r["predicted"])}</span>'\
            f'<div style="display:flex;align-items:center;gap:7px;justify-content:flex-end;"><div style="flex:1;max-width:40px;height:4px;background:var(--line2);border-radius:2px;overflow:hidden;"><div style="height:100%;width:{barw:.0f}%;background:{col};"></div></div>'\
            f'<span class="mono" style="font-size:12px;font-weight:600;color:{col};min-width:42px;text-align:right;">{gap}</span></div></div>'
    return rows

# ════════ 01 BARGAIN ════════
with t_bar:
    st.markdown(f"""<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;font-size:12.5px;color:var(--muted);">
      <span style="display:inline-flex;align-items:center;gap:8px;"><span style="width:9px;height:9px;border-radius:2px;background:var(--pos);"></span><span style="color:var(--text);font-weight:600;">割安</span>＝ AI予測より実価格が低い（実力比で低評価）</span>
      <span style="display:inline-flex;align-items:center;gap:8px;"><span style="width:9px;height:9px;border-radius:2px;background:var(--neg);"></span><span style="color:var(--text);font-weight:600;">割高</span>＝ AI予測より実価格が高い（名声・人気で上振れ）</span></div>""", unsafe_allow_html=True)
    filt = df[df[TARGET] >= 500_000].copy()
    bargains = filt.nlargest(8, "gap_ratio")
    overpriced = filt.nsmallest(8, "gap_ratio")
    head = '<div class="mono" style="display:grid;grid-template-columns:24px 1fr 42px 30px 62px 62px 92px;gap:10px;padding:9px 16px;font-size:9.5px;letter-spacing:.08em;color:var(--faint);text-transform:uppercase;"><span>#</span><span>PLAYER</span><span>POS</span><span>AGE</span><span style="text-align:right;">実価格</span><span style="text-align:right;">AI予測</span><span style="text-align:right;">乖離</span></div>'
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'{panel_open()}overflow:hidden;"><div style="display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--line);"><div style="display:flex;align-items:center;gap:9px;"><span style="width:3px;height:16px;background:var(--pos);border-radius:2px;"></span><span style="font-weight:600;font-size:14px;">割安 TOP 8</span></div><span class="mono" style="font-size:10px;letter-spacing:.14em;color:var(--pos);padding:3px 8px;background:var(--pos-dim);border-radius:4px;">UNDERVALUED</span></div>{head}{table_rows(bargains, True)}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'{panel_open()}overflow:hidden;"><div style="display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid var(--line);"><div style="display:flex;align-items:center;gap:9px;"><span style="width:3px;height:16px;background:var(--neg);border-radius:2px;"></span><span style="font-weight:600;font-size:14px;">割高 TOP 8</span></div><span class="mono" style="font-size:10px;letter-spacing:.14em;color:var(--neg);padding:3px 8px;background:var(--neg-dim);border-radius:4px;">OVERVALUED</span></div>{head}{table_rows(overpriced, False)}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="mono" style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:2px;">バリュエーション・マップ</div><div class="mono" style="font-size:11px;color:var(--faint);margin-bottom:8px;">X: 実際の市場価値 (log) · Y: AI予測価値 (log) · 対角線より上＝割安</div>', unsafe_allow_html=True)
    smp = filt.sample(min(1600, len(filt)), random_state=42)
    fig = px.scatter(smp, x=TARGET, y="predicted", log_x=True, log_y=True,
                     color="gap_ratio", range_color=[-2, 2],
                     color_continuous_scale=[[0, NEG], [0.5, "#6b7280"], [1, POS]],
                     hover_data={"name": True})
    mn, mx = smp[TARGET].min(), smp[TARGET].max()
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                  line=dict(color="rgba(255,255,255,0.25)", dash="dash", width=1.3)))
    fig.update_traces(marker=dict(size=6))
    style_fig(fig, 380); fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ════════ 02 PREDICT ════════
with t_pred:
    mode = st.radio("MODE", ["既存選手で照合", "カスタム入力"], horizontal=True, label_visibility="collapsed")
    if mode == "既存選手で照合":
        c1, c2 = st.columns([1.1, 1])
        with c1:
            names = sorted(df["name"].dropna().unique())
            sel = st.selectbox("SELECT PLAYER", names)
            p = df[df["name"] == sel].iloc[0]
            gr = p["gap_ratio"] * 100
            col = POS if gr >= 0 else NEG
            st.markdown(f"""<div style="margin-top:6px;">
              <div style="display:flex;align-items:baseline;gap:12px;"><div style="font-size:22px;font-weight:700;">{esc(p['name'])}</div><div class="mono" style="font-size:12px;color:var(--muted);">{esc(p['team'])}</div></div>
              <div class="mono" style="display:flex;gap:8px;margin-top:12px;font-size:11px;"><span style="padding:4px 10px;border:1px solid var(--line2);border-radius:5px;color:var(--muted);">POS <span style="color:var(--text);">{esc(p['position'])}</span></span><span style="padding:4px 10px;border:1px solid var(--line2);border-radius:5px;color:var(--muted);">AGE <span style="color:var(--text);">{int(p['age'])}歳</span></span></div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;margin-top:18px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden;">
                <div style="background:var(--panel);padding:16px 18px;"><div class="mono" style="font-size:10px;letter-spacing:.12em;color:var(--faint);text-transform:uppercase;margin-bottom:8px;">実際の市場価値</div><div class="mono" style="font-size:24px;font-weight:600;">{fmt_eur(p[TARGET])}</div></div>
                <div style="background:var(--panel);padding:16px 18px;"><div class="mono" style="font-size:10px;letter-spacing:.12em;color:var(--faint);text-transform:uppercase;margin-bottom:8px;">AI 予測価値</div><div class="mono" style="font-size:24px;font-weight:600;color:var(--accent);">{fmt_eur(p['predicted'])}</div></div>
              </div></div>""", unsafe_allow_html=True)
        with c2:
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=p["predicted"]/1e6,
                number={"suffix": "M", "prefix": "€", "font": {"color": ACCENT, "size": 40, "family": MONO}},
                gauge={"axis": {"range": [0, max(p[TARGET], p["predicted"])/1e6*1.25], "tickcolor": "#56575d"},
                       "bar": {"color": ACCENT, "thickness": 0.32},
                       "bgcolor": "rgba(255,255,255,0.04)", "borderwidth": 0,
                       "threshold": {"line": {"color": "#94a3b8", "width": 2}, "thickness": 0.8, "value": p[TARGET]/1e6}}))
            g.update_layout(height=230, paper_bgcolor=PAPER, font=dict(color="#cfd0d4", family=MONO), margin=dict(l=20, r=20, t=14, b=0))
            st.plotly_chart(g, use_container_width=True)
            verdict = "割安 UNDERVALUED" if gr >= 0 else "割高 OVERVALUED"
            st.markdown(f'<div style="text-align:center;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--inset);"><span class="mono" style="font-size:13px;font-weight:600;color:{col};">{verdict}　{"+" if gr>=0 else ""}{gr:.1f}%</span></div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            age = st.slider("年齢", 16, 40, 23)
            gpg = st.slider("1試合あたりゴール", 0.0, 1.2, 0.45, 0.01)
            apg = st.slider("1試合あたりアシスト", 0.0, 0.8, 0.22, 0.01)
            mpg = st.slider("1試合あたり出場時間（分）", 0, 90, 76)
            pos = st.selectbox("ポジション", sorted(df["position"].unique()))
        custom = {"age": age, "height": 182, "appearance": 60,
                  "goals": gpg*60, "assists": apg*60, "minutes played": mpg*60,
                  "yellow cards": 0.1, "red cards": 0.0, "second yellow cards": 0.0,
                  "goals conceded": 0.0, "clean sheets": 0.0, "days_injured": 0,
                  "games_injured": 0, "award": 0, "winger": 0,
                  "position": pos, "team": sorted(df["team"].unique())[0],
                  "goals_per_game": gpg, "assists_per_game": apg,
                  "minutes_per_game": mpg, "injury_rate": 0.0}
        X_in = transform_single(custom, encoders, feature_names)
        pred = predict_value(model, X_in)
        with c2:
            st.markdown(f"""<div style="border:1px solid var(--line);border-radius:12px;background:linear-gradient(160deg,var(--panel),var(--inset));padding:30px 22px;text-align:center;">
              <div class="mono" style="font-size:10px;letter-spacing:.18em;color:var(--faint);text-transform:uppercase;">ESTIMATED FAIR VALUE</div>
              <div class="mono" style="font-size:50px;font-weight:700;color:var(--accent);margin:16px 0 6px;letter-spacing:-.02em;text-shadow:0 0 28px var(--accent-dim);">{fmt_eur(pred)}</div>
              <div class="mono" style="font-size:14px;color:var(--muted);">{fmt_jpy(pred)}</div>
              <div style="margin-top:18px;font-size:12px;color:var(--faint);line-height:1.6;">スライダーを動かすと学習済みXGBoostモデルがリアルタイムに再計算します。</div></div>""", unsafe_allow_html=True)

# ════════ 03 ANALYSIS ════════
def html_bars(pairs, grad, unit="M"):
    mx = max(v for _, v in pairs)
    out = ""
    for lab, v in pairs:
        out += f'<div style="display:grid;grid-template-columns:130px 1fr 58px;gap:12px;align-items:center;margin-bottom:10px;">'\
               f'<span class="mono" style="font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{esc(lab)}</span>'\
               f'<div style="height:9px;background:var(--inset);border-radius:5px;overflow:hidden;"><div style="height:100%;width:{v/mx*100:.1f}%;background:{grad};border-radius:5px;"></div></div>'\
               f'<span class="mono" style="font-size:12px;text-align:right;">€{v:.1f}{unit}</span></div>'
    return out

with t_ana:
    c1, c2 = st.columns(2)
    with c1:
        pos_med = (df.groupby("position")[TARGET].median()/1e6).sort_values(ascending=False)
        bars = [(k, float(v)) for k, v in pos_med.items()]
        st.markdown(f'{panel_open()}padding:20px;"><div style="font-weight:600;font-size:14px;margin-bottom:2px;">ポジション別 市場価値（中央値）</div><div class="mono" style="font-size:10.5px;color:var(--faint);margin-bottom:16px;">MEDIAN VALUE BY POSITION</div>{html_bars(bars, "linear-gradient(90deg,rgba(214,162,58,.5),var(--accent))")}</div>', unsafe_allow_html=True)
    with c2:
        team_mean = (df.groupby("team")[TARGET].mean()/1e6).sort_values(ascending=False).head(10)
        bars = [(k, float(v)) for k, v in team_mean.items()]
        st.markdown(f'{panel_open()}padding:20px;"><div style="font-weight:600;font-size:14px;margin-bottom:2px;">チーム別 平均市場価値 TOP 10</div><div class="mono" style="font-size:10.5px;color:var(--faint);margin-bottom:16px;">MEAN VALUE BY CLUB</div>{html_bars(bars, "linear-gradient(90deg,rgba(72,192,122,.4),var(--pos))")}</div>', unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-weight:600;font-size:14px;">年齢 × 市場価値の分布</div><div class="mono" style="font-size:10.5px;color:var(--faint);margin-bottom:8px;">X: AGE · Y: MARKET VALUE (log)</div>', unsafe_allow_html=True)
    smp = df.sample(min(2500, len(df)), random_state=0)
    fig = px.scatter(smp, x="age", y=TARGET, color="position", log_y=True, opacity=.55, hover_data=["name", "team"])
    fig.update_traces(marker=dict(size=6))
    style_fig(fig, 360)
    st.plotly_chart(fig, use_container_width=True)

# ════════ 04 MODEL ════════
with t_mod:
    c1, c2 = st.columns(2)
    with c1:
        imp = feature_importance(model, feature_names)
        items = list(imp.items())[:12]
        imx = items[0][1]
        rows = ""
        for lab, v in items:
            rows += f'<div style="display:grid;grid-template-columns:120px 1fr 50px;gap:12px;align-items:center;margin-bottom:9px;">'\
                    f'<span class="mono" style="font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{esc(lab)}</span>'\
                    f'<div style="height:8px;background:var(--inset);border-radius:4px;overflow:hidden;"><div style="height:100%;width:{v/imx*100:.1f}%;background:var(--accent);border-radius:4px;"></div></div>'\
                    f'<span class="mono" style="font-size:11.5px;color:var(--muted);text-align:right;">{v:.3f}</span></div>'
        st.markdown(f'{panel_open()}padding:20px;"><div style="font-weight:600;font-size:14px;margin-bottom:2px;">特徴量重要度 TOP 12</div><div class="mono" style="font-size:10.5px;color:var(--faint);margin-bottom:16px;">FEATURE IMPORTANCE</div>{rows}</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div style="font-weight:600;font-size:14px;">予測精度（実際 vs AI予測）</div><div class="mono" style="font-size:10.5px;color:var(--faint);margin-bottom:8px;">対角線に近いほど高精度</div>', unsafe_allow_html=True)
        X_test, y_test, y_pred_log = splits
        perf = pd.DataFrame({"actual": np.expm1(y_test), "predicted": np.expm1(y_pred_log)})
        perf["gr"] = (perf["predicted"] - perf["actual"]) / perf["actual"]
        ps = perf.sample(min(1400, len(perf)), random_state=1)
        fig = px.scatter(ps, x="actual", y="predicted", log_x=True, log_y=True, opacity=.5,
                         color="gr", range_color=[-2, 2],
                         color_continuous_scale=[[0, NEG], [0.5, "#6b7280"], [1, POS]])
        mn, mx = ps["actual"].min(), ps["actual"].max()
        fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(color=ACCENT, dash="dash", width=1.5)))
        fig.update_traces(marker=dict(size=5))
        style_fig(fig, 250); fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:var(--line);border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:8px;">
          <div style="background:var(--panel);padding:14px;"><div class="mono" style="font-size:9.5px;color:var(--faint);text-transform:uppercase;margin-bottom:6px;">CV R²</div><div class="mono" style="font-size:19px;font-weight:600;color:var(--accent);">{metrics['cv_r2_mean']:.3f}</div></div>
          <div style="background:var(--panel);padding:14px;"><div class="mono" style="font-size:9.5px;color:var(--faint);text-transform:uppercase;margin-bottom:6px;">TEST R²</div><div class="mono" style="font-size:19px;font-weight:600;">{metrics['test_r2']:.3f}</div></div>
          <div style="background:var(--panel);padding:14px;"><div class="mono" style="font-size:9.5px;color:var(--faint);text-transform:uppercase;margin-bottom:6px;">MAE</div><div class="mono" style="font-size:19px;font-weight:600;">{fmt_eur(metrics['test_mae_eur'])}</div></div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    cards = [("ALGORITHM", "XGBoost Regressor", "n_estimators=1000 · max_depth=7"),
             ("TARGET", "current_value（log1p変換）", "右裾の歪みを補正"),
             ("FEATURES", f"{len(feature_names)}個（基本+派生+カテゴリ）", "per-game 正規化を採用")]
    cc = "".join(f'<div style="border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:14px 18px;flex:1;min-width:200px;"><div class="mono" style="font-size:10px;letter-spacing:.12em;color:var(--accent);margin-bottom:7px;">{a}</div><div style="font-size:13px;">{b}</div><div style="font-size:11px;color:var(--muted);margin-top:3px;">{c}</div></div>' for a, b, c in cards)
    st.markdown(f'<div style="display:flex;gap:10px;flex-wrap:wrap;">{cc}</div>', unsafe_allow_html=True)

# ── フッター ────────────────────────────────────────────────────────────────
st.markdown("""<div class="mono" style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-top:34px;padding-top:18px;border-top:1px solid var(--line);font-size:10.5px;color:var(--faint);">
  <span>DATA · Transfermarkt / Kaggle &nbsp;|&nbsp; MODEL · XGBoost（実推論） &nbsp;|&nbsp; 教育・ポートフォリオ目的</span>
  <div style="display:flex;gap:7px;"><span style="padding:3px 9px;border:1px solid var(--line);border-radius:4px;">Python</span><span style="padding:3px 9px;border:1px solid var(--line);border-radius:4px;">scikit-learn</span><span style="padding:3px 9px;border:1px solid var(--line);border-radius:4px;">XGBoost</span><span style="padding:3px 9px;border:1px solid var(--line);border-radius:4px;">Plotly</span></div>
</div>""", unsafe_allow_html=True)
