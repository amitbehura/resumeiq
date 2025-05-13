from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
import os
import json
import re
import fitz  # PyMuPDF

# === Load environment variables ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# === FastAPI setup ===
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# === CORS (for frontend development) ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Serve HTML ===
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# =====================
# === JD Extraction ===
# =====================
class JDText(BaseModel):
    text: str

@app.post("/extract-jd")
async def extract_jd(file: UploadFile = File(...)):
    pdf = fitz.open(stream=await file.read(), filetype="pdf")
    max_pages = 2
    text = "".join([pdf[i].get_text() for i in range(min(len(pdf), max_pages))])
    return await extract_keywords(text)

@app.post("/extract-jd-text")
async def extract_jd_text(data: JDText):
    return await extract_keywords(data.text[:3000])

async def extract_keywords(text: str):
    prompt = f"""
You are an AI recruiter assistant. Given the Job Description below, perform only these steps:

Step 1: Extract 5–7 keyword groups. Each group should contain 1–3 related keywords or synonyms used to describe the same skill (e.g., ["Product Manager", "Product Management"]).

Step 2: From the JD only, extract exclusion phrases — e.g., "No freshers", "Not open to relocation", etc. Only include exclusions if explicitly mentioned. Do not guess.

Return the result in this exact JSON format:
{{
  "groups": [["keyword1", "keyword2"], ["keyword3"], ...],
  "exclude": ["exclusion1", "exclusion2"]
}}

IMPORTANT:
- No assumptions. Extract exact phrases only from the JD.
- Max 7 groups, and 3 keywords per group.
- Only respond with valid JSON.

Job Description:
{text.strip()[:3000]}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You extract keywords and exclusions from job descriptions."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()

        if not content:
            return {"groups": [], "exclude": [], "error": "GPT returned empty response."}

        try:
            parsed = json.loads(content)
            return {
                "groups": parsed.get("groups", []),
                "exclude": parsed.get("exclude", [])
            }
        except json.JSONDecodeError as e:
            print("❌ JSON parsing error:", e)
            print("🚨 GPT Raw Content:", content)
            return {"groups": [], "exclude": [], "error": f"Invalid JSON format from GPT: {str(e)}"}

    except Exception as e:
        print("🔥 GPT API Error:", e)
        return {"groups": [], "exclude": [], "error": str(e)}

# ================================
# === Boolean Query Generator ===
# ================================
class BooleanRequest(BaseModel):
    groups: List[List[str]]
    exclude: List[str]

@app.post("/generate-boolean")
async def generate_boolean(data: BooleanRequest):
    boolean_parts = []

    for group in data.groups:
        if group:
            terms = ' OR '.join(f'"{kw}"' for kw in group)
            boolean_parts.append(f"({terms})")

    boolean_query = ' AND '.join(boolean_parts)

    if data.exclude:
        not_part = ' OR '.join(f'"{kw}"' for kw in data.exclude)
        boolean_query += f' AND NOT ({not_part})'

    return {"boolean_query": boolean_query}

# =============================
# === Resume Match & Rewriter ===
# =============================
@app.post("/generate-pointers")
async def generate_pointers(
    resume: UploadFile = File(...),
    target_match: int = Form(...),
    jd_text: Optional[str] = Form("")
):
    import math

    pdf = fitz.open(stream=await resume.read(), filetype="pdf")
    max_pages = 2
    resume_text = "".join([pdf[i].get_text() for i in range(min(len(pdf), max_pages))])

    keyword_prompt = f"""
Extract 10–15 of the most important keywords from the Job Description below that a resume should include. These should be noun phrases or action terms — no explanations.

Respond with a valid Python list of strings only.

Job Description:
{jd_text.strip()[:3000]}
"""

    try:
        keyword_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract keywords only from the JD."},
                {"role": "user", "content": keyword_prompt}
            ]
        )
        keyword_text = keyword_response.choices[0].message.content.strip()
        all_keywords = eval(keyword_text) if keyword_text.startswith("[") else []
    except Exception as e:
        return {"match_score": 0, "updated_pointers": [], "error": f"Keyword extraction failed: {str(e)}"}

    match_percent = max(50, min(target_match, 100))
    k = max(1, math.ceil((match_percent / 100) * len(all_keywords)))
    selected_keywords = all_keywords[:k]
    selected_keywords_text = ", ".join(selected_keywords)

    prompt = f"""
You are a professional resume assistant AI trusted by hiring managers.

## STEP 1: Classify the JD
Based on the Job Description below, classify it using these three fields:

ROLE TITLE: Most likely role title (e.g., Product Manager, HRBP, Software Engineer)  
ROLE TYPE: One of ["Strategic", "Technical", "Mixed", "Support", "Analytical", "People-facing"]  
LEVEL: One of ["Entry-Level", "IC", "Team Lead", "Manager", "Director", "VP"]  

## STEP 2: Rewrite Resume Bullets
1. Rewrite or enhance up to 10 bullet points from the resume using only real experience.
2. The slider is set to a match target of **{target_match}%**.
   - If it is 100%, you must include **all** of the following {k} keywords exactly once.  
   - If it is less than 100%, try to include **approximately that % of keywords** (each at most once).
3. Here are the keywords:  
{selected_keywords_text}
4. Tailor tone, metrics, and structure to fit the ROLE TYPE and LEVEL.

Guidelines:
- Focus on measurable outcomes, strong verbs, and clean bullet structure.
- Do NOT fabricate experience or include irrelevant info.
- Do NOT mention “JD”, “aligned to role”, or use generic filler.
- Do NOT include educational qualifications or certifications in bullets.

---
🧠 JD:
{jd_text.strip()[:3000]}

📄 Resume:
{resume_text.strip()[:3000]}

---
🎯 OUTPUT FORMAT (strict):
Match Score: <score>%  
ROLE TITLE: <...>  
ROLE TYPE: <...>  
LEVEL: <...>  

• Bullet 1  
• Bullet 2  
...(max 10)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You rewrite resumes based on job descriptions."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content.strip()

        match = re.search(r"Match Score:\s*(\d{1,3})", result)
        score = int(match.group(1)) if match else 70

        role_title = re.search(r"ROLE TITLE:\s*(.+)", result)
        role_type = re.search(r"ROLE TYPE:\s*(.+)", result)
        level = re.search(r"LEVEL:\s*(.+)", result)

        role_title_val = role_title.group(1).strip() if role_title else "Unknown"
        role_type_val = role_type.group(1).strip() if role_type else "Unknown"
        level_val = level.group(1).strip() if level else "Unknown"

        pointers = [
            line.strip("\u2022- \n")
            for line in result.splitlines()
            if line.strip().startswith("\u2022")
        ]

        used_keywords = [
            kw for kw in selected_keywords
            if any(kw.lower() in p.lower() for p in pointers)
        ]
        missed_keywords = list(set(selected_keywords) - set(used_keywords))

        return {
            "match_score": score,
            "role_title": role_title_val,
            "role_type": role_type_val,
            "level": level_val,
            "updated_pointers": pointers[:10],
            "keywords_used": used_keywords,
            "keywords_missed": missed_keywords
        }

    except Exception as e:
        print("⚠️ GPT Resume Rewrite Error:", e)
        return {
            "match_score": 0,
            "updated_pointers": [],
            "error": str(e)
        }
