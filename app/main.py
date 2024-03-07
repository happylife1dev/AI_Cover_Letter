# curl -X GET "http://127.0.0.1:8000/" -H "x-token: <TOKEN>"


import asyncio
import json
import time
import os

from fastapi import Form, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, File, UploadFile, Form
from fastapi import Request, Depends
# from utils import do_match, handle_resume_upload, get_clean_text_from_url, generate_response, structurize_with_gpt
from fastapi.responses import JSONResponse
from fastapi import HTTPException, status, Header
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import PyPDF2
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

CV_RESULTS_TABLE = "cv_results"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Fallback value is for local development, you can remove it if you want.
SECRET_TOKEN = os.environ.get("API_SECRET_TOKEN", "default_fallback_token")


class CV_Result_Item(BaseModel):
    created_at: str
    analyzer_response: dict
    job: str
    original_file: str
    analysis_status: str


def get_token_header(x_token: str = Header(None)):
    print(x_token)
    if x_token is None:
        print("Token header is missing!")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token header is missing",
        )
    if x_token != SECRET_TOKEN:
        print(f"Invalid token received: {x_token}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return x_token


# def get_token_header(x_token: str = Header(...)):  # Rename x_token to Authorization or any other header you want.

#     if x_token != SECRET_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid token",
#         )
#     return x_token


app = FastAPI()
# app = FastAPI(debug=False, docs_url=None)
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_PATH = "/tmp"


class WelcomeResponse(BaseModel):
    welcome_message: Optional[str]
    error_message: Optional[str]


# fast api status method
@app.get("/status")
async def status():
    return {"status": "ok"}


@app.get("/cv_result")
async def get_alldata_cv_result():
    data = supabase.table(CV_RESULTS_TABLE).select("*").execute()
    return data.data


@app.get("/cv_result/{data_id}")
async def get_data_cv_result(data_id: str):
    data = supabase.table(CV_RESULTS_TABLE).select("*").eq("id", data_id).execute()
    return data.data


@app.post("/cv_result/{data_id}")
async def updata_cv_result(data_id: str, cv_result_item: CV_Result_Item):
    data = supabase.table(CV_RESULTS_TABLE).update(cv_result_item.dict()).eq("id", data_id).execute()
    return data.data


# create method that sleeps 2 seocnd then returns a test object as json
@app.post("/mock_analyse_resume")
async def analyse_resume_mockup(
        resume: UploadFile = File(...),
        jobbeschreibung: str = Form(...),
        token: str = Form(...)

):
    TEST = {'title': 'Analysis Report for IM A. SAMPLE II', 'requiredSkills': [{
        'skill': 'Architekturmanagement und Erstellung der IT-Lösungs-Architekturen im Rahmen des Leistungskontextes (in verschiedenen Vorgehensmodellen wie agil oder Wasserfall) unter Berücksichtigung bestehender (Unternehmens-)Standards, Nachhaltigkeit, Wirtschaftlichkeit, Marktfähigkeit',
        'experience': 'IM A. SAMPLE II does not have experience in Architekturmanagement und Erstellung der IT-Lösungs-Architekturen.',
        'references': [], 'match': 'no'}, {
        'skill': 'Beratung des Auftraggebers zur technischen Architektur(strategie) der IT-Systeme im Themenumfeld sowie deren Weiterentwicklung und aufzeigen von Verbesserungsmöglichkeiten der bestehenden technischen Architektur',
        'experience': 'IM A. SAMPLE II does not have experience in Beratung des Auftraggebers zur technischen Architektur(strategie) der IT-Systeme.',
        'references': [], 'match': 'no'}, {
        'skill': 'Fördern und Treiben der Umsetzung der technologischen Strategie, des architektonisch-technischen Wissensaustauschs und der Innovation und Standardisierung im Projekt',
        'experience': 'IM A. SAMPLE II does not have experience in Fördern und Treiben der Umsetzung der technologischen Strategie, des architektonisch-technischen Wissensaustauschs und der Innovation und Standardisierung im Projekt.',
        'references': [
        ], 'match': 'no'}, {
        'skill': 'Durchführung des Risikomanagements durch Bewertung und Dokumentation von technischen Risiken, Ableiten von und Beraten zu Handlungsoptionen zur Risikobeseitigung oder Minimierung',
        'experience': 'IM A. SAMPLE II does not have experience in Durchführung des Risikomanagements durch Bewertung und Dokumentation von technischen Risiken, Ableiten von und Beraten zu Handlungsoptionen zur Risikobeseitigung oder Minimierung.',
        'references': [], 'match': 'no'}, {
        'skill': 'Beratung bei der Umsetzung der technischen Architektur in der Softwareentwicklung',
        'experience': 'IM A. SAMPLE II does not have experience in Beratung bei der Umsetzung der technischen Architektur in der Softwareentwicklung.',
        'references': [], 'match': 'no'}],
            'niceToHaveSkills': [], 'goodMatching': False,
            'explanation': "IM A. SAMPLE II does not meet the required skills for the job. The candidate's background is in Business Administration with Marketing Emphasis, while the job requires technical skills in IT and software development."}

    await asyncio.sleep(20)
    return JSONResponse(content=TEST)


# @app.get("/welcome") #TODO disabled save in cache for single user
def get_welcome_message(token: str = Depends(get_token_header)):
    return WelcomeResponse(welcome_message=f'Welcome there ', error_message=None)
