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


def get_embedding(text: str) -> np.ndarray:
    """Gemini REST APIでテキストをベクトル化する。"""
    text = text[:1000]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={API_KEY}"
    payload = json.dumps({
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
        "taskType": "SEMANTIC_SIMILARITY",
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as res:
        data = json.loads(res.read())
    return np.array(data["embedding"]["values"])


def find_missing_skills(job_skills: list[str], career_text: str) -> list[str]:
    career_lower = career_text.lower()
    return [skill for skill in job_skills if skill.lower() not in career_lower]


def generate_advice(job_text: str, career_text: str, missing_skills: list[str], score: float) -> str:
    """Gemini REST APIで改善アドバイスを生成。"""
    prompt = f"""
あなたはキャリアアドバイザーです。
以下の情報を基に、応募者が求人に採用されるための具体的な改善アドバイスを日本語で作成してください。

【求人情報】
{job_text[:800]}

【職務経歴・スキル】
{career_text[:800]}

【マッチングスコア】
類似度: {score:.0%}

【不足スキル】
{", ".join(missing_skills) if missing_skills else "特になし"}

以下の形式で回答してください：
1. 総評（2〜3文）
2. 優先的に補強すべきスキル（箇条書き・具体的なアクション付き）
3. アピールできる強み（職務経歴から読み取れるもの）
4. 応募に向けた次のステップ（3つ）
"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}]
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            data = json.loads(res.read())
        return data["candidates"][0]["content"]["parts"][0]["text"]
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


# ── エンドポイント ──

@app.get("/")
def health():
    return {"status": "ok", "message": "Job Matcher API is running"}


@app.post("/match", response_model=MatchResult)
def match(req: MatchRequest):
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

    all_job_skills = req.job.required_skills + req.job.preferred_skills
    missing = find_missing_skills(all_job_skills, career_text)
    advice = generate_advice(job_text, career_text, missing, score)

    return MatchResult(
        score_sbert=round(score, 4),
        score_w2v=None,
        missing_skills=missing,
        advice=advice,
    )


@app.post("/parse-csv")
async def parse_csv(file: UploadFile = File(...)):
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
