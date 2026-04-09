from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import csv
import io
import time
import numpy as np

from google import genai
from google.genai import types

app = FastAPI(title="Job Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Gemini API 設定
api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
print(f"API key loaded: {bool(api_key)}")
client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})

print("Job Matcher API ready.")


# ── ユーティリティ ──

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_embedding(text: str) -> np.ndarray:
    text = text[:1000]
    result = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return np.array(result.embeddings[0].values)


def find_missing_skills(job_skills: list[str], career_text: str) -> list[str]:
    career_lower = career_text.lower()
    return [skill for skill in job_skills if skill.lower() not in career_lower]


def generate_advice(job_text: str, career_text: str, missing_skills: list[str], score: float) -> str:
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
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
        )
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
