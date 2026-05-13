# ⚽ サッカー選手 市場価値分析・予測

Transfermarktの選手データ（10,754名）を用いて、選手の市場価値を機械学習で予測するWebアプリ。
**XGBoost** によるモデル化と **Streamlit** によるインタラクティブな分析ダッシュボードを提供する。

## 主な機能

| タブ | 内容 |
|---|---|
| 📈 概要 | データ統計・市場価値TOP20選手・価値分布 |
| 🔍 探索分析 | 年齢/ポジション/チーム別の価値比較 |
| 🤖 選手予測 | 既存選手の予測値・カスタム入力での予測 |
| 💎 割安/割高選手 | モデル予測と実際の価値の乖離から、割安・割高選手を抽出 |
| 🎯 モデル解釈 | 特徴量重要度・予測精度の可視化 |

## モデル

| 項目 | 内容 |
|---|---|
| アルゴリズム | XGBoost (n=500, depth=6) |
| 評価 | 5-fold 交差検証 + 80/20 ホールドアウト |
| 目的変数 | `current_value`（log1p 変換） |
| 特徴量 | 21個（基本15 + 派生4 + カテゴリ2） |

### 派生特徴量
モデル精度向上のため、以下の特徴量を追加で生成：

- `goals_per_game` — 1試合あたりゴール
- `assists_per_game` — 1試合あたりアシスト
- `minutes_per_game` — 1試合あたり出場時間
- `injury_rate` — 怪我による試合欠場率

## 工夫した点

1. **対数変換**：市場価値は分布の右裾が極端に長いため、`log1p` 変換で誤差を抑制
2. **派生特徴量**：合計値ではなく per-game に正規化することで、出場数の違いを吸収
3. **5-fold CV**：単一の train/test 分割に依存しない汎化性能評価
4. **割安/割高分析**：モデル予測と実際の乖離から、市場で見落とされた選手を抽出する応用例を実装

## セットアップ

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

データは `data/final_data.csv` に配置。出典：[Kaggle: Football Players Transfermarkt Dataset](https://www.kaggle.com/datasets/davidcariboo/player-scores)

## 技術スタック

- **Python 3.14**
- **scikit-learn** / **XGBoost** — 機械学習
- **pandas** / **numpy** — データ処理
- **Streamlit** — UI
- **Plotly** — インタラクティブ可視化

## 今後の改善点

- [ ] SHAP値による個別予測の説明可能性
- [ ] LightGBM / CatBoost との比較
- [ ] 時系列での市場価値トレンド分析
- [ ] チーム・リーグ集約特徴量の追加
