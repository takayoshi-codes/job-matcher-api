from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import csv
import io
import time
import json
import numpy as np
import urllib.request
import urllib.error

app = FastAPI(title="Job Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
print(f"API key loaded: {bool(API_KEY)}")
print("Job Matcher API ready.")


# ── ユーティリティ ──

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def call_gemini(url: str, payload: dict) -> dict:
    """Gemini REST APIを呼び出す共通関数。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read())


def get_embedding(text: str) -> np.ndarray:
    """Gemini Embedding APIでテキストをベクトル化する。"""
    text = text[:1000]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={API_KEY}"
    result = call_gemini(url, {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": text}]},
        "taskType": "SEMANTIC_SIMILARITY",
    })
    return np.array(result["embedding"]["values"])


def generate_gemini_text(prompt: str) -> str:
    """Gemini APIでテキストを生成する。複数モデルにフォールバック。"""
    models = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            result = call_gemini(url, {
                "contents": [{"parts": [{"text": prompt}]}]
            })
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            continue
    return "アドバイス生成に失敗しました。"


def find_missing_skills(job_skills: list[str], career_text: str) -> list[str]:
    career_lower = career_text.lower()
    return [skill for skill in job_skills if skill.lower() not in career_lower]


def generate_full_analysis(
    job_text: str,
    career_text: str,
    missing_skills: list[str],
    score: float,
) -> str:
    """Gemini APIで総合分析を生成。"""
    prompt = f"""
あなたはキャリアアドバイザーです。
以下の情報を基に、応募者への総合的なキャリアアドバイスを日本語で作成してください。

【求人情報】
{job_text[:600]}

【職務経歴・スキル】
{career_text[:600]}

【マッチングスコア】
類似度: {score:.0%}

【不足スキル】
{", ".join(missing_skills) if missing_skills else "特になし"}

以下の形式で回答してください：

## 総評
（2〜3文で全体的な評価）

## 優先的に補強すべきスキル
（箇条書き・具体的なアクション付き。不足スキルがない場合はさらなるレベルアップの提案）

## アピールできる強み
（職務経歴から読み取れる強みを3つ）

## 応募に向けた次のステップ
（具体的なアクション3つ）

## この経験・スキルで応募できる求人タイプ
（具体的に4〜5個。案件の種類・規模・業界を含めて）

## おすすめの副業プラットフォーム
（スキルに合ったプラットフォームを2〜3個、理由付きで）
"""
    return generate_gemini_text(prompt)


def generate_job_suggestions(career_text: str) -> str:
    """職務経歴だけから応募可能な求人タイプを提案する。"""
    prompt = f"""
以下の職務経歴・スキルを持つエンジニアが副業・フリーランスとして応募できる求人タイプを分析してください。

【職務経歴・スキル】
{career_text[:800]}

以下の形式で日本語で回答してください：

## あなたが応募できる求人タイプ
（具体的に5〜6個、各タイプに期待単価と理由を含めて）

## 特に強くアピールできる分野
（3つ、理由付きで）

## おすすめの副業プラットフォーム
（3つ、それぞれの特徴と向いている理由を含めて）

## 今すぐ応募できるレベルか？
（正直な評価と、もし準備が必要なら何が必要か）
"""
    return generate_gemini_text(prompt)


# ── スキーマ ──

class JobInput(BaseModel):
    title: str = ""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    description: str = ""


class CareerInput(BaseModel):
    name: str = ""
    skills: str = ""
    summary_consulting: str = ""
    summary_management: str = ""
    summary_it: str = ""
    projects: str = ""


class MatchRequest(BaseModel):
    job: JobInput
    career: CareerInput


class MatchResult(BaseModel):
    score_sbert: float
    score_w2v: float | None
    missing_skills: list[str]
    advice: str
    job_suggestions: str


class CareerAnalysisRequest(BaseModel):
    career: CareerInput


class CareerAnalysisResult(BaseModel):
    suggestions: str


# ── エンドポイント ──

@app.get("/")
def health():
    return {"status": "ok", "message": "Job Matcher API is running"}


@app.post("/match", response_model=MatchResult)
def match(req: MatchRequest):
    """職務経歴と求人票のマッチングスコアを算出する。"""
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

    # Embedding でベクトル化（リトライあり）
    score = 0.0
    for attempt in range(3):
        try:
            job_vec = get_embedding(job_text)
            career_vec = get_embedding(career_text)
            score = cosine_similarity(job_vec, career_vec)
            break
        except Exception as e:
            if attempt == 2:
                raise HTTPException(status_code=500, detail=f"Embedding エラー: {str(e)}")
            time.sleep(2)

    # 不足スキル検出
    all_job_skills = req.job.required_skills + req.job.preferred_skills
    missing = find_missing_skills(all_job_skills, career_text)

    # 総合分析・アドバイス生成
    advice = generate_full_analysis(job_text, career_text, missing, score)

    # 応募可能求人タイプの提案
    job_suggestions = generate_job_suggestions(career_text)

    return MatchResult(
        score_sbert=round(score, 4),
        score_w2v=None,
        missing_skills=missing,
        advice=advice,
        job_suggestions=job_suggestions,
    )


@app.post("/career-analysis", response_model=CareerAnalysisResult)
def career_analysis(req: CareerAnalysisRequest):
    """求人票なしで職務経歴だけから応募可能な求人タイプを分析する。"""
    career_text = " ".join([
        req.career.skills,
        req.career.summary_consulting,
        req.career.summary_management,
        req.career.summary_it,
        req.career.projects,
    ]).strip()

    if not career_text:
        raise HTTPException(status_code=400, detail="職務経歴が空です")

    suggestions = generate_job_suggestions(career_text)
    return CareerAnalysisResult(suggestions=suggestions)


@app.post("/parse-csv")
async def parse_csv(file: UploadFile = File(...)):
    """Career Builder が出力したCSVを受け取り、CareerInput形式に変換する。"""
    try:
        content = await file.read()
        decoded = content.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(decoded))
        rows = list(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSVパースエラー: {str(e)}")

    data: dict[str, str] = {}
    for row in rows[1:]:
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
