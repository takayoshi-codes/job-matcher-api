# job-matcher-api

求人票 × 職務経歴書 マッチング診断の FastAPI バックエンド。

## 技術スタック

- **FastAPI** — REST API
- **sentence-transformers** — BERT ベクトル化
- **gensim** — Word2Vec ベクトル化
- **MeCab** — 日本語形態素解析
- **Google Gemini API** — 改善アドバイス生成
- **Railway** — デプロイ先

## エンドポイント

| メソッド | パス | 説明 |
|---|---|---|
| GET | `/` | ヘルスチェック |
| POST | `/match` | マッチングスコア算出 |
| POST | `/parse-csv` | Career BuilderのCSVを解析 |

## 環境変数

| 変数名 | 説明 |
|---|---|
| `GEMINI_API_KEY` | Google Gemini APIキー |

## ローカル起動

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
