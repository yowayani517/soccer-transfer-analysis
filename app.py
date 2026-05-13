import os, sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from src.preprocess import load_and_clean, build_features, transform_single, TARGET
from src.model import train, predict_value, predict_all, feature_importance

# ── ページ設定 ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="⚽ 移籍市場分析", page_icon="⚽", layout="wide",
                   initial_sidebar_state="collapsed")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "final_data.csv")

# ── カスタムCSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* 全体背景 */
  .stApp { background: #0f1117; color: #e0e0e0; }

  /* ヒーローバナー */
  .hero {
    background: linear-gradient(135deg, #1a1f3a 0%, #0d2137 50%, #0a3d2e 100%);
    border-radius: 16px; padding: 40px 48px; margin-bottom: 24px;
    border: 1px solid #2a3a5c;
  }
  .hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0 0 8px;
             background: linear-gradient(90deg, #38bdf8, #34d399);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .hero p  { color: #94a3b8; font-size: 1.05rem; margin: 0; }

  /* ガイドカード */
  .guide-card {
    background: #1e2640; border-radius: 12px; padding: 20px;
    border-left: 4px solid #38bdf8; margin-bottom: 12px;
  }
  .guide-card h4 { color: #38bdf8; margin: 0 0 6px; font-size: 1rem; }
  .guide-card p  { color: #94a3b8; margin: 0; font-size: 0.9rem; }

  /* メトリクスカード */
  .metric-card {
    background: #1e2640; border-radius: 12px; padding: 20px 24px;
    border: 1px solid #2a3a5c; text-align: center;
  }
  .metric-label { color: #64748b; font-size: 0.8rem; text-transform: uppercase;
                  letter-spacing: .05em; margin-bottom: 6px; }
  .metric-value { color: #f0f4ff; font-size: 1.7rem; font-weight: 700; }
  .metric-sub   { color: #38bdf8; font-size: 0.85rem; margin-top: 4px; }

  /* バッジ */
  .badge-green { background:#064e3b; color:#34d399; border-radius:6px;
                 padding:3px 10px; font-size:.8rem; font-weight:600; }
  .badge-red   { background:#450a0a; color:#f87171; border-radius:6px;
                 padding:3px 10px; font-size:.8rem; font-weight:600; }
  .badge-blue  { background:#0c2246; color:#38bdf8; border-radius:6px;
                 padding:3px 10px; font-size:.8rem; font-weight:600; }

  /* セクションタイトル */
  .section-title {
    font-size: 1.3rem; font-weight: 700; color: #f0f4ff;
    padding-bottom: 8px; border-bottom: 2px solid #2a3a5c; margin: 28px 0 16px;
  }

  /* ステップバッジ */
  .step { display:inline-block; background:#1d4ed8; color:#fff;
          border-radius:50%; width:24px; height:24px; line-height:24px;
          text-align:center; font-weight:700; font-size:.85rem; margin-right:8px; }

  /* タブ強調 */
  .stTabs [role="tab"] { font-size: .95rem; font-weight: 600; }
  .stTabs [aria-selected="true"] { color: #38bdf8 !important; border-bottom-color: #38bdf8 !important; }

  /* データフレーム */
  .stDataFrame { border-radius: 10px; overflow: hidden; }

  /* ボタン */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0ea5e9);
    color: white; border: none; border-radius: 10px;
    font-weight: 700; font-size: 1rem; padding: 12px 0;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .85; }

  /* サイドバー非表示 */
  [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── ユーティリティ ──────────────────────────────────────────────────────────
def fmt_eur(v: float) -> str:
    if v >= 1e6: return f"€{v/1e6:.1f}M"
    if v >= 1e3: return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

def fmt_jpy(v: float) -> str:
    m = v / 1e6 * 160
    if m >= 10000: return f"約{m/10000:.0f}億円"
    return f"約{m:.0f}百万円"

# ── データ・モデル読み込み ─────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_clean(DATA_PATH)

@st.cache_resource
def get_model(_df):
    X, y, encoders, names = build_features(_df)
    model, metrics, splits = train(X, y)
    preds = predict_all(model, X)
    return model, encoders, names, metrics, splits, preds

if not os.path.exists(DATA_PATH):
    st.error("❌ `data/final_data.csv` が見つかりません。")
    st.stop()

df = get_data()
with st.spinner("⚙️ AIモデルを学習中... 初回のみ30〜60秒かかります"):
    model, encoders, feature_names, metrics, splits, all_preds = get_model(df)

df["predicted"] = all_preds
df["gap"]       = df["predicted"] - df[TARGET]
df["gap_ratio"] = df["gap"] / df[TARGET]

# ── ヒーローバナー ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>⚽ サッカー移籍市場 AI分析</h1>
  <p>10,754選手のデータから市場価値を予測・コスパ選手を自動発掘するダッシュボード</p>
</div>
""", unsafe_allow_html=True)

# ── 使い方ガイド ───────────────────────────────────────────────────────────
with st.expander("📖 使い方ガイド（初めての方はここを読んでください）", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("""<div class="guide-card">
        <h4><span class="step">1</span>💎 コスパ選手タブ</h4>
        <p>AIが「実力のわりに安い選手」を自動でランキング表示。まずここを見よう。</p>
    </div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="guide-card">
        <h4><span class="step">2</span>🔮 移籍金予測タブ</h4>
        <p>好きな選手を選ぶと「AIが算出した適正価格」と比較できる。</p>
    </div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="guide-card">
        <h4><span class="step">3</span>📊 データ分析タブ</h4>
        <p>ポジション・年齢・チーム別のグラフを眺めてトレンドを把握。</p>
    </div>""", unsafe_allow_html=True)
    c4.markdown("""<div class="guide-card">
        <h4><span class="step">4</span>🤖 モデル精度タブ</h4>
        <p>AIの予測精度と「価値を決める要因TOP10」を確認できる。</p>
    </div>""", unsafe_allow_html=True)

# ── KPIカード ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📈 データ概要</p>', unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
cards = [
    ("分析対象選手", f"{len(df):,}人", "Transfermarktデータ"),
    ("平均市場価値", fmt_eur(df[TARGET].mean()), fmt_jpy(df[TARGET].mean())),
    ("最高市場価値", fmt_eur(df[TARGET].max()), df.loc[df[TARGET].idxmax(), "name"]),
    ("モデル精度 R²", f"{metrics['cv_r2_mean']:.3f}", f"±{metrics['cv_r2_std']:.3f} (5-fold CV)"),
    ("予測誤差 MAE", fmt_eur(metrics["test_mae_eur"]), "テストデータ平均誤差"),
]
for col, (label, val, sub) in zip([k1,k2,k3,k4,k5], cards):
    col.markdown(f"""<div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── タブ ──────────────────────────────────────────────────────────────────
tab_bargain, tab_predict, tab_analysis, tab_model = st.tabs([
    "💎 コスパ選手ランキング",
    "🔮 移籍金を予測する",
    "📊 データ分析",
    "🤖 AIモデルについて",
])

# ════════════════════════════════════════════════════════════════
# 💎 コスパ選手ランキング
# ════════════════════════════════════════════════════════════════
with tab_bargain:
    st.markdown("""
    <p style="color:#94a3b8; font-size:.95rem; margin-bottom:20px;">
    🟢 <b>コスパ良い（割安）</b> = AIの予測価格より <b>実際の市場価格が安い</b> 選手。実力のわりに低評価。<br>
    🔴 <b>割高</b> = AIの予測価格より <b>実際の市場価格が高い</b> 選手。名声や人気で価格が吊り上がっている可能性。
    </p>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    filt = df[df[TARGET] >= 500_000].copy()

    with col_left:
        st.markdown('<p class="section-title">🟢 コスパ最高！割安選手 TOP20</p>', unsafe_allow_html=True)
        bargains = filt.nlargest(20, "gap_ratio")[
            ["name","team","position","age",TARGET,"predicted","gap_ratio"]
        ].copy()
        bargains["gap_ratio_fmt"] = bargains["gap_ratio"].apply(lambda x: f"+{x*100:.0f}%")
        bargains[TARGET]      = bargains[TARGET].apply(fmt_eur)
        bargains["predicted"] = bargains["predicted"].apply(fmt_eur)
        bargains = bargains.rename(columns={
            "name":"選手名","team":"チーム","position":"ポジション","age":"年齢",
            TARGET:"実際","predicted":"AI予測","gap_ratio_fmt":"割安度"
        }).drop(columns=["gap_ratio"])
        st.dataframe(bargains, width="stretch", hide_index=True,
                     column_config={"割安度": st.column_config.TextColumn("割安度 🟢")})

    with col_right:
        st.markdown('<p class="section-title">🔴 要注意！割高選手 TOP20</p>', unsafe_allow_html=True)
        overpriced = filt.nsmallest(20, "gap_ratio")[
            ["name","team","position","age",TARGET,"predicted","gap_ratio"]
        ].copy()
        overpriced["gap_ratio_fmt"] = overpriced["gap_ratio"].apply(lambda x: f"{x*100:.0f}%")
        overpriced[TARGET]      = overpriced[TARGET].apply(fmt_eur)
        overpriced["predicted"] = overpriced["predicted"].apply(fmt_eur)
        overpriced = overpriced.rename(columns={
            "name":"選手名","team":"チーム","position":"ポジション","age":"年齢",
            TARGET:"実際","predicted":"AI予測","gap_ratio_fmt":"割高度"
        }).drop(columns=["gap_ratio"])
        st.dataframe(overpriced, width="stretch", hide_index=True,
                     column_config={"割高度": st.column_config.TextColumn("割高度 🔴")})

    st.markdown('<p class="section-title">📍 全選手の割安・割高マップ</p>', unsafe_allow_html=True)
    st.caption("横軸=実際の市場価値 / 縦軸=AIが予測した市場価値 / 対角線より上 = 割安・下 = 割高")
    sample = filt.sample(min(2000, len(filt)), random_state=42)
    fig_map = px.scatter(
        sample, x=TARGET, y="predicted",
        log_x=True, log_y=True,
        color="gap_ratio",
        color_continuous_scale=[[0,"#ef4444"],[0.5,"#6b7280"],[1,"#22c55e"]],
        range_color=[-2, 2],
        hover_data={"name": True, "team": True, "position": True,
                    TARGET: ":,.0f", "predicted": ":,.0f", "gap_ratio": ":.2f"},
        labels={TARGET:"実際の市場価値 (€)", "predicted":"AI予測 (€)", "gap_ratio":"割安度"},
        template="plotly_dark",
    )
    mn = min(sample[TARGET].min(), sample["predicted"].min())
    mx = max(sample[TARGET].max(), sample["predicted"].max())
    fig_map.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
        line=dict(color="#94a3b8", dash="dash", width=1.5), name="実際=予測ライン",
        showlegend=True))
    fig_map.update_layout(height=480, margin=dict(l=0,r=0,t=10,b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#141824",
                          coloraxis_colorbar=dict(title="割安度"))
    st.plotly_chart(fig_map, width="stretch")

# ════════════════════════════════════════════════════════════════
# 🔮 移籍金予測
# ════════════════════════════════════════════════════════════════
with tab_predict:
    pred_tab1, pred_tab2 = st.tabs(["👤 既存選手で確認", "✏️ カスタム入力で予測"])

    with pred_tab1:
        st.markdown("**好きな選手を選ぶと、実際の市場価値とAIの予測を比較できます。**")
        player_names = sorted(df["name"].dropna().unique())
        sel = st.selectbox("選手名を検索・選択", player_names,
                           help="名前の一部を入力すると絞り込めます")
        if sel:
            p = df[df["name"] == sel].iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("実際の市場価値",  fmt_eur(p[TARGET]),  fmt_jpy(p[TARGET]))
            c2.metric("AIの予測価格",    fmt_eur(p["predicted"]), fmt_jpy(p["predicted"]))
            gap_pct = p["gap_ratio"] * 100
            label = f"{'✅ 割安' if gap_pct > 0 else '⚠️ 割高'} {abs(gap_pct):.1f}%"
            c3.metric("評価", label)

            st.markdown("<hr style='border-color:#2a3a5c'>", unsafe_allow_html=True)
            i1, i2, i3, i4 = st.columns(4)
            i1.markdown(f"**チーム**<br>{p['team']}", unsafe_allow_html=True)
            i2.markdown(f"**ポジション**<br>{p['position']}", unsafe_allow_html=True)
            i3.markdown(f"**年齢**<br>{int(p['age'])} 歳", unsafe_allow_html=True)
            i4.markdown(f"**身長**<br>{p['height']:.0f} cm", unsafe_allow_html=True)

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=p["predicted"] / 1e6,
                delta={"reference": p[TARGET] / 1e6, "suffix":"M差"},
                title={"text":"AI予測価格 (€M)"},
                gauge={"axis":{"range":[0, max(p[TARGET], p["predicted"]) / 1e6 * 1.3]},
                       "bar":{"color":"#38bdf8"},
                       "steps":[{"range":[0, p[TARGET]/1e6], "color":"#1e293b"}],
                       "threshold":{"line":{"color":"#f59e0b","width":3},
                                    "thickness":.75, "value":p[TARGET]/1e6}},
                number={"suffix":"M €", "font":{"size":36}},
            ))
            fig_gauge.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)",
                                    font={"color":"#e0e0e0"}, margin=dict(l=20,r=20,t=40,b=0))
            st.plotly_chart(fig_gauge, width="stretch")

    with pred_tab2:
        st.markdown("**選手のスタッツを入力すると、AIが適正市場価値を計算します。**")
        with st.form("custom_form"):
            row1 = st.columns(4)
            row2 = st.columns(4)
            row3 = st.columns(4)
            custom = {
                "age":               row1[0].number_input("年齢", 15, 45, 24),
                "height":            row1[1].number_input("身長 (cm)", 150, 215, 180),
                "appearance":        row1[2].number_input("出場数", 0, 600, 60),
                "goals":             row1[3].number_input("ゴール数", 0, 400, 15),
                "assists":           row2[0].number_input("アシスト数", 0, 300, 8),
                "minutes played":    row2[1].number_input("出場時間 (分)", 0, 60000, 4500),
                "yellow cards":      row2[2].number_input("イエロー/試合", 0.0, 1.0, 0.1, 0.01),
                "red cards":         row2[3].number_input("レッド/試合", 0.0, 0.5, 0.0, 0.01),
                "second yellow cards": 0.0,
                "goals conceded":    row3[0].number_input("失点/試合 (GK用)", 0.0, 5.0, 0.0, 0.1),
                "clean sheets":      row3[1].number_input("クリーンシート率 (GK用)", 0.0, 1.0, 0.0, 0.01),
                "days_injured":      row3[2].number_input("怪我日数", 0, 2000, 0),
                "games_injured":     row3[3].number_input("怪我試合数", 0, 200, 0),
                "award": 0, "winger": 0,
                "position": st.selectbox("ポジション", sorted(df["position"].unique())),
                "team":     st.selectbox("チーム", sorted(df["team"].unique())),
            }
            custom["goals_per_game"]   = custom["goals"]  / max(custom["appearance"], 1)
            custom["assists_per_game"] = custom["assists"] / max(custom["appearance"], 1)
            custom["minutes_per_game"] = custom["minutes played"] / max(custom["appearance"], 1)
            custom["injury_rate"]      = custom["games_injured"] / max(custom["appearance"] + custom["games_injured"], 1)
            submitted = st.form_submit_button("💰 市場価値を予測する", use_container_width=True)

        if submitted:
            X_in = transform_single(custom, encoders, feature_names)
            pred = predict_value(model, X_in)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#064e3b,#0c2246);border-radius:14px;
                        padding:28px;text-align:center;margin-top:16px;">
              <p style="color:#94a3b8;margin:0 0 8px;font-size:.9rem;">AIが算出した適正市場価値</p>
              <p style="color:#34d399;font-size:2.6rem;font-weight:800;margin:0;">{fmt_eur(pred)}</p>
              <p style="color:#38bdf8;font-size:1.1rem;margin:8px 0 0;">{fmt_jpy(pred)}</p>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 📊 データ分析
# ════════════════════════════════════════════════════════════════
with tab_analysis:
    a1, a2 = st.columns(2)
    with a1:
        st.markdown('<p class="section-title">📌 ポジション別 市場価値中央値</p>', unsafe_allow_html=True)
        pos_df = (df.groupby("position")[TARGET].median()
                  .sort_values(ascending=True).reset_index())
        fig_pos = px.bar(pos_df, x=TARGET, y="position", orientation="h",
                         template="plotly_dark",
                         color=TARGET, color_continuous_scale="teal",
                         labels={TARGET:"市場価値中央値 (€)", "position":"ポジション"})
        fig_pos.update_layout(showlegend=False, height=340,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#141824",
                              coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_pos, width="stretch")

    with a2:
        st.markdown('<p class="section-title">📌 チーム別 平均市場価値 TOP15</p>', unsafe_allow_html=True)
        team_df = (df.groupby("team")[TARGET].mean()
                   .sort_values(ascending=False).head(15)
                   .sort_values().reset_index())
        fig_team = px.bar(team_df, x=TARGET, y="team", orientation="h",
                          template="plotly_dark",
                          color=TARGET, color_continuous_scale="blues",
                          labels={TARGET:"平均市場価値 (€)", "team":"チーム"})
        fig_team.update_layout(showlegend=False, height=340,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#141824",
                               coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_team, width="stretch")

    st.markdown('<p class="section-title">📌 年齢 × 市場価値（ポジション別）</p>', unsafe_allow_html=True)
    st.caption("選手にホバーすると詳細が表示されます")
    fig_age = px.scatter(
        df.sample(min(3000, len(df)), random_state=0),
        x="age", y=TARGET, color="position",
        log_y=True, opacity=.55, size_max=8,
        hover_data=["name","team"],
        template="plotly_dark",
        labels={"age":"年齢", TARGET:"市場価値 (€)"},
    )
    fig_age.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#141824",
                          margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_age, width="stretch")

# ════════════════════════════════════════════════════════════════
# 🤖 モデルについて
# ════════════════════════════════════════════════════════════════
with tab_model:
    m1, m2 = st.columns([1, 1])
    with m1:
        st.markdown('<p class="section-title">🏆 特徴量重要度（価値を左右する要因 TOP12）</p>', unsafe_allow_html=True)
        st.caption("AIが「市場価値を予測するのに重要だ」と判断した項目のランキング")
        imp = feature_importance(model, feature_names)
        imp_df = pd.DataFrame(list(imp.items()), columns=["特徴量","重要度"]).head(12)
        fig_imp = px.bar(imp_df, x="重要度", y="特徴量", orientation="h",
                         template="plotly_dark",
                         color="重要度", color_continuous_scale="teal")
        fig_imp.update_yaxes(autorange="reversed")
        fig_imp.update_layout(showlegend=False, height=380,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#141824",
                              coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_imp, width="stretch")

    with m2:
        st.markdown('<p class="section-title">🎯 予測精度（実際 vs AI予測）</p>', unsafe_allow_html=True)
        st.caption("点が対角線に近いほど精度が高い。色は割安（緑）/割高（赤）を表す")
        X_test, y_test, y_pred_log = splits
        perf_df = pd.DataFrame({
            "actual":    np.expm1(y_test),
            "predicted": np.expm1(y_pred_log),
        })
        perf_df["gap_r"] = (perf_df["predicted"] - perf_df["actual"]) / perf_df["actual"]
        fig_perf = px.scatter(
            perf_df.sample(min(1500, len(perf_df)), random_state=1),
            x="actual", y="predicted",
            log_x=True, log_y=True, opacity=.5,
            color="gap_r",
            color_continuous_scale=[[0,"#ef4444"],[0.5,"#6b7280"],[1,"#22c55e"]],
            range_color=[-2,2],
            template="plotly_dark",
            labels={"actual":"実際 (€)","predicted":"AI予測 (€)","gap_r":"割安度"},
        )
        mn = perf_df["actual"].min(); mx = perf_df["actual"].max()
        fig_perf.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
            line=dict(color="#f59e0b", dash="dash", width=2), name="完全一致"))
        fig_perf.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#141824",
                               coloraxis_showscale=False, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_perf, width="stretch")

        st.markdown(f"""
        <div style="background:#1e2640;border-radius:10px;padding:16px 20px;">
        <table style="width:100%;color:#e0e0e0;border-collapse:collapse;">
          <tr><td>🎯 交差検証 R²</td><td style="text-align:right;color:#34d399;font-weight:700;">{metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}</td></tr>
          <tr><td>📊 テスト R²</td><td style="text-align:right;color:#38bdf8;font-weight:700;">{metrics['test_r2']:.3f}</td></tr>
          <tr><td>📉 予測誤差 MAE</td><td style="text-align:right;color:#f59e0b;font-weight:700;">{fmt_eur(metrics['test_mae_eur'])}</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)
