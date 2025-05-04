from fastapi import FastAPI, File, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.vectorstores import InMemoryVectorStore
import openai
import json
from io import BytesIO
from pathlib import Path
from typing_extensions import TypedDict


# ----------- JSON schema for guard‑railed output ----------------------------
class SummaryJson(TypedDict):
    content: str
    title: str
    description: str
    location: str


# ----------- Helper function to get embeddings ----------------------------
def get_embeddings(text: str, model="text-embedding-ada-002"):
    """
    Gets the embeddings for the provided text using OpenAI's embeddings API.
    """
    response = openai.embeddings.create(
        model=model, input=[text]  # The input should be a list of texts
    )
    embeddings = [embedding["embedding"] for embedding in response["data"]]
    return np.array(embeddings)


# ----------- main callable --------------------------------------------------
def pdf_to_summary_json(pdf_file: BytesIO, model_name: str = "gpt-4") -> SummaryJson:
    """
    Extracts text from `pdf_file`, processes it, and returns mock test questions in SummaryJson format.
    """
    # 1) Load & chunk the PDF content using PyPDFLoader and RecursiveCharacterTextSplitter
    pdf_path = Path("uploaded_pdf.pdf")
    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    pages = PyPDFLoader(str(pdf_path)).load()

    # Use RecursiveCharacterTextSplitter to break the document into chunks
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1_000, chunk_overlap=150
    ).split_documents(pages)

    # 2) Embed & store chunks in RAM
    vectordb = InMemoryVectorStore(OpenAIEmbeddings())
    vectordb.add_documents(chunks)

    # 3) Retrieve top‑k context based on a query
    question = "Make a test for students based on the lecture material"
    docs = vectordb.similarity_search(question, k=6)
    context = "\n\n".join(d.page_content for d in docs)

    # 4) Generate JSON with structured output using GPT
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    structured_llm = llm.with_structured_output(SummaryJson)  # forces valid schema

    prompt = f"""You are a mock quiz generator. Based on the provided lecture notes, create the following types of questions:
    - One answer with 4 options to pick.
    - Multiple choice with 5-6 options to pick.
    - Matching questions.

    The provided lecture content: {context}

    Output the result in the following JSON format:
    [
      {{
        "title": "Who was the first president of the United States?",  // One choice question
        "variants": [
          "George Washington",  // Option 1
          "Abraham Lincoln"  // Option 2
        ],
        "correctVariantIndex": 0  // Correct answer is the first option (index 0)
      }},
      {{"title": "Which of the following countries were part of the Allied Powers in World War II?", 
        "variants": [
          "United States", "Germany", "Soviet Union", "Italy"
        ],
        "correctVariantIndex": [0, 2]
      }},
      {{
        "title": "Match the famous historical figures to their corresponding country:", 
        "variants": [
          "George Washington", "Albert Einstein", "Winston Churchill", "Mahatma Gandhi"
        ],
        "correctVariantIndex": [0, 1, 2, 3],
        "matchWith": [
          "United States", "Germany", "United Kingdom", "India"
        ]
      }}
    ]

    Ensure that the quiz questions are based only on the provided lecture notes or the test will fail."""

    json_out = structured_llm.invoke(prompt)

    return json_out


# ----------- FastAPI setup --------------------------------------------------
app = FastAPI()


@app.post("/generate-quiz/")
async def generate_quiz(file: UploadFile = File(...)):
    """
    Endpoint to generate mock quiz questions from the uploaded PDF file.
    """
    # Read the PDF file content
    pdf_file = await file.read()
    try:
        # Process the PDF and generate the summary JSON
        result = pdf_to_summary_json(BytesIO(pdf_file))
        return json.dumps(result, indent=2, ensure_ascii=False)
    except Exception as e:
        return {"error": str(e)}
