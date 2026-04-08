from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import csv
import io
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from gensim.models import KeyedVectors
from gensim.downloader import load as gensim_load
import MeCab

app = FastAPI(title="Job Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデルのロード（起動時に1回だけ）
print("Loading models...")
sbert_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

# Word2Vec（日本語学習済みモデル）
# 初回はダウンロードに時間がかかります
try:
    w2v_model = gensim_load("glove-wiki-gigaword-100")  # 英語fallback
except:
    w2v_model = None

# MeCab（日本語形態素解析）
try:
    mecab = MeCab.Tagger("-Owakati")
except:
    mecab = None

# Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
gemini = genai.GenerativeModel("gemini-1.5-flash")

print("Models loaded.")


# ── ユーティリティ ──

def tokenize_ja(text: str) -> list[str]:
    """MeCabで分かち書き。失敗時はスペース分割。"""
    if mecab:
        try:
            return mecab.parse(text).strip().split()
        except:
            pass
    return text.split()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sbert_encode(text: str) -> np.ndarray:
    return sbert_model.encode(text, normalize_embeddings=True)


def w2v_encode(text: str) -> np.ndarray | None:
    if not w2v_model:
        return None
    tokens = tokenize_ja(text)
    vecs = [w2v_model[t] for t in tokens if t in w2v_model]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def find_missing_skills(job_skills: list[str], career_text: str) -> list[str]:
    """求人スキルのうち、職務経歴に含まれないものを返す。"""
    career_lower = career_text.lower()
    missing = []
    for skill in job_skills:
        # 完全一致 or 部分一致で確認
        if skill.lower() not in career_lower:
            missing.append(skill)
    return missing


def generate_advice(
    job_text: str,
    career_text: str,
    missing_skills: list[str],
    score_sbert: float,
    score_w2v: float | None,
) -> str:
    """Gemini APIで改善アドバイスを生成。"""
    prompt = f"""
あなたはキャリアアドバイザーです。
以下の情報を基に、応募者が求人に採用されるための具体的な改善アドバイスを日本語で作成してください。

【求人情報】
{job_text[:800]}

【職務経歴・スキル】
{career_text[:800]}

【マッチングスコア】
BERT類似度: {score_sbert:.0%}
{"Word2Vec類似度: " + f"{score_w2v:.0%}" if score_w2v is not None else ""}

【不足スキル】
{", ".join(missing_skills) if missing_skills else "特になし"}

以下の形式で回答してください：
1. 総評（2〜3文）
2. 優先的に補強すべきスキル（箇条書き・具体的なアクション付き）
3. アピールできる強み（職務経歴から読み取れるもの）
4. 応募に向けた次のステップ（3つ）
"""
    try:
        response = gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"アドバイス生成エラー: {str(e)}"


# ── スキーマ ──

class JobInput(BaseModel):
    title: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    description: str = ""


class CareerInput(BaseModel):
    name: str = ""
    skills: str = ""          # 技術スタック（カンマ区切り）
    summary_consulting: str = ""
    summary_management: str = ""
    summary_it: str = ""
    projects: str = ""        # 職務経歴テキスト


class MatchRequest(BaseModel):
    job: JobInput
    career: CareerInput


class MatchResult(BaseModel):
    score_sbert: float
    score_w2v: float | None
    missing_skills: list[str]
    advice: str


# ── エンドポイント ──

@app.get("/")
def health():
    return {"status": "ok", "message": "Job Matcher API is running"}


@app.post("/match", response_model=MatchResult)
def match(req: MatchRequest):
    """職務経歴と求人票のマッチングスコアを算出する。"""

    # テキストを結合
    job_text = " ".join([
        req.job.title,
        " ".join(req.job.required_skills),
        " ".join(req.job.preferred_skills),
        req.job.description,
    ]).strip()

    career_text = " ".join([
        req.career.skills,
        req.career.summary_consulting,
        req.career.summary_management,
        req.career.summary_it,
        req.career.projects,
    ]).strip()

    if not job_text or not career_text:
        raise HTTPException(status_code=400, detail="求人票または職務経歴が空です")

    # BERT スコア
    job_vec_sbert = sbert_encode(job_text)
    career_vec_sbert = sbert_encode(career_text)
    score_sbert = cosine_similarity(job_vec_sbert, career_vec_sbert)

    # Word2Vec スコア
    score_w2v = None
    job_vec_w2v = w2v_encode(job_text)
    career_vec_w2v = w2v_encode(career_text)
    if job_vec_w2v is not None and career_vec_w2v is not None:
        score_w2v = cosine_similarity(job_vec_w2v, career_vec_w2v)

    # 不足スキル
    all_job_skills = req.job.required_skills + req.job.preferred_skills
    missing = find_missing_skills(all_job_skills, career_text)

    # Gemini アドバイス
    advice = generate_advice(job_text, career_text, missing, score_sbert, score_w2v)

    return MatchResult(
        score_sbert=round(score_sbert, 4),
        score_w2v=round(score_w2v, 4) if score_w2v is not None else None,
        missing_skills=missing,
        advice=advice,
    )


@app.post("/parse-csv")
async def parse_csv(file: UploadFile = File(...)):
    """Career Builder が出力したCSVを受け取り、CareerInput形式に変換する。"""
    try:
        content = await file.read()
        decoded = content.decode("utf-8-sig")  # BOM対応
        reader = csv.reader(io.StringIO(decoded))
        rows = list(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSVパースエラー: {str(e)}")

    data: dict[str, str] = {}
    for row in rows[1:]:  # ヘッダー行をスキップ
        if len(row) >= 3:
            section, key, value = row[0], row[1], row[2]
            data[f"{section}_{key}"] = value

    career = CareerInput(
        name=data.get("基本情報_氏名", ""),
        skills=", ".join(filter(None, [
            data.get("技術スタック_言語", ""),
            data.get("技術スタック_FW", ""),
            data.get("技術スタック_DB", ""),
            data.get("技術スタック_クラウド", ""),
            data.get("技術スタック_AI/ML", ""),
            data.get("技術スタック_ツール", ""),
        ])),
        summary_consulting=data.get("スキルサマリ_コンサルスキル", ""),
        summary_management=data.get("スキルサマリ_マネジメントスキル", ""),
        summary_it=data.get("スキルサマリ_ITスキル", ""),
        projects=" ".join(filter(None, [
            data.get(k, "") for k in data if "職務経歴" in k and "業務内容" in k
        ])),
    )

    return career.model_dump()
