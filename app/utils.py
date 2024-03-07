# TODO  speed up by extracting resume in structure and job beore sending to gpt4


import re
from bs4 import BeautifulSoup
from pyppeteer import launch
import uuid
import time
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi import Request
from langchain.prompts import ChatPromptTemplate
import json
from prompts_json import json_schema, system_message_content_without_coverletter, system_message_structurize_json
from langchain.chains.openai_functions import create_structured_output_chain
import asyncio
import concurrent.futures
import threading

from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import PyPDF2
import os
import langchain
from langchain.cache import InMemoryCache
from langchain.cache import SQLiteCache


from pathlib import Path
from typing import Optional

from langchain.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from typing import Any, Dict, List
from langchain.schema import LLMResult, HumanMessage
# load env variables
from dotenv import load_dotenv
load_dotenv()

# llm = ChatOpenAI(model='gpt-4', temperature=0.1, max_tokens=2000, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=["\n\n", "Human:", "System:"])
# llm = ChatOpenAI(model='gpt-3.5-turbo-16k', temperature=0.1)

#
llm = ChatOpenAI(model='gpt-4-0613', temperature=0.1)
# llm = ChatOpenAI(model='gpt-3.5-turbo-16k-0613', temperature=0.1) # TODO change back to GPT-4


def get_pdf_content(pdf_path):
    pdf = fitz.open(pdf_path)

    text = ""
    for page in pdf:
        text += page.get_text()

    # TODO disable OCR because of package size
    # if not text.strip():
    #     reader = easyocr.Reader(['en'])
    #     for page in pdf:
    #         pix = page.get_pixmap()
    #         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #         text += ' '.join([t[1] for t in reader.readtext(np.array(img))])

    return text


def highlight_words_in_pdf(pdf_path, words_to_highlight):
    pdf = fitz.open(pdf_path)

    for word in words_to_highlight.split(","):
        word = word.strip()
        for page in pdf:
            text_instances = page.search_for(word)
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)

    temp_output_path = "/tmp/highlighted_output.pdf"
    pdf.save(temp_output_path)

    return temp_output_path


def extract_text_from_pdf(file_path):

    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text


def handle_resume_upload(uploaded_file, resume_path):
    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.filename,
                        "FileType": uploaded_file.content_type}
        if file_details["FileType"] == "application/pdf":
            file_path = resume_path / f"resume_{uuid.uuid4()}.pdf"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.file.read())

            return extract_text_from_pdf(file_path), None
        else:
            return None, "Please upload a valid PDF file."
    return None, None


async def structurize_with_gpt(text, model_name='gpt-3.5-turbo-16k-0613', system_message_content=system_message_structurize_json):
    global system_message_structurize_json
    response = await generate_response(
        system_message_content, text, model_name)
    return response



async def do_match(resume, job_desc):
    global llm, system_message_content_without_coverletter, json_schema, system_message_structurize_json

    # langchain.llm_cache = InMemoryCache()
    # We can do the same thing with a SQLite cache

    #TODO 
    langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

    # count number of words for resume and job description and print it
    # start timer  for processing time from now until the end of the function

    start = time.time()

    print(" Lenth before structurize:")
    # counter number of words in resume and job description            

    print("Length of resume: " + str(len(resume)))
    print("Length of job description: " + str(len(job_desc)))

    # job_desc = structurize_with_gpt(job_desc)
    # resume = structurize_with_gpt(resume)

    print(" Lenth after structurize:")
    print("Length of resume: " + str(len(resume)))
    print("Length of job description: " + str(len(job_desc)))

    system_message = SystemMessage(
        content=system_message_content_without_coverletter)
    human_message = HumanMessage(
        content=f"Resume:\n{resume}\njob description:\n{job_desc}")
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message,
            human_message
        ])
    print(prompt)

    # structurize is good but makes it extremly slow with gpt4

    # chain = create_structured_output_chain(json_schema, llm, prompt, verbose=True)
    chain = LLMChain(llm=llm, prompt=prompt)
    output = await chain.arun({})

    end = time.time()
    # print processing time in seconds
    print("Processing time: " + str(end - start))

    # output = json.loads(output)

    # get md first

    # convert to json

    # print (output)

    # # now convert to json
    # system_message = SystemMessage(content=system_message_convert_json)
    # human_message = HumanMessage( content=f"{output}")
    # prompt = ChatPromptTemplate.from_messages(
    # [
    #     system_message,
    #     human_message
    # ] )
    # # print (prompt)
    # # chain = create_structured_output_chain(json_schema, llm, prompt, verbose=True)
    # chain = LLMChain(llm=llm, prompt=prompt)
    # output = chain.run({})

    return output

    # messages = [system_message, human_message]
    # result = llm(messages)

    # return result.content


# URLS


async def get_page_content(url):
    browser = await launch(handleSIGINT=False, handleSIGTERM=False, handleSIGHUP=False)
    page = await browser.newPage()
    await page.setViewport({'width': 1366, 'height': 768})

    SAFARI_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15"
    await page.setUserAgent(SAFARI_USER_AGENT)

    try:
        await page.goto(url, waitUntil="domcontentloaded", timeout=60000)
    except Exception as e:  # It's better to catch a general exception for simplicity here.
        print(f"Error: {e}")
        await browser.close()
        return None

    content = await page.content()
    await browser.close()
    return content


def get_clean_text_from_url(url):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread, url)

    # call openai api to extract the relevant parts related to a job desciription and remove the rest
        clean_page = future.result()

        # if number of characters is greater than 3000: summarize the page
        # print size of the clean page message
        extracted_job = clean_page
        print("Length of clean page: " + str(len(clean_page)))
        if len(clean_page) > 3000:
            system_message_content = "You summarize a given page and extract the part related to the job description. Dont make up anything, just extract the relevant parts."
            response = generate_response(
                system_message_content, clean_page, "gpt-3.5-turbo-16k")
            extracted_job = response
            print("Length of clean page after Extraction: " +
                  str(len(extracted_job)))
        # print size of the extracted job
        return extracted_job


def run_in_thread(url):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(get_clean_text_from_url_async(url))
    finally:
        loop.close()


async def get_clean_text_from_url_async(url):
    content = await get_page_content(url)

    if not content:
        return None  # return None if there was an error

    soup = BeautifulSoup(content, 'html.parser')

    for script in soup(['script', 'style']):
        script.decompose()

    clean_text = soup.get_text()
    clean_text = re.sub(r'\n+', '\n', clean_text).strip()

    return clean_text

# Usage example:
# text_content = get_clean_text_from_url("https://www.example.com")
# print(text_content)


class MyCustomSyncHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Sync handler being called in a `thread_pool_executor`: token: {token}")


class MyCustomAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when chain starts running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        class_name = serialized["name"]
        print("Hi! I just woke up. Your llm is starting")

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        print("zzzz....")
        await asyncio.sleep(0.3)
        print("Hi! I just woke up. Your llm is ending")

async def generate_response(system_message_content: str, human_message_content: str, model_name: str = 'gpt-3.5-turbo') -> AIMessage:
    """
    Generates a response based on the given system and human messages.

    Args:
    - system_message_content (str): The content of the system message.
    - human_message_content (str): The content of the human message.
    - model_name (str): The name of the model to use. Defaults to 'gpt-4'.

    Returns:
    - AIMessage: The response generated by the LLM.
    """

    llm = ChatOpenAI(model=model_name, callbacks=[MyCustomSyncHandler(), MyCustomAsyncHandler()])

    # Create SystemMessage
    system_message = SystemMessage(content=system_message_content)

    # Create HumanMessage
    human_message = HumanMessage(content=human_message_content)

    # Create messages list
    messages = [system_message, human_message]
    result  = await llm.agenerate([messages])
    result = result.generations[0][0].text

    print (result)
    # result = llm(messages)

    return result
