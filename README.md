# Customer Support Agent

A small, local customer-support assistant that answers questions from company FAQs and policies. It uses **LangChain** for the model, prompt, document splitting, and local embeddings, and **LangGraph** to route each question through retrieval, answering, or a human handoff.

The first version is intentionally limited: it reads documents and answers questions. It has no tools to access orders, issue refunds, or change customer accounts.

## What it can do

- Read PDF, DOCX, Markdown, and TXT files.
- Answer from uploaded files and show the source filename and PDF page where available.
- Mark questions for a human when the documents do not support an answer.
- Let an operator download a handoff note.
- Run with local models through Ollama; no model API key is required.

Uploads and document vectors stay in the app session's memory and are cleared when the session ends. This is a local pilot, not a multi-customer hosted service.

## Local model choice

Install [Ollama for Windows](https://docs.ollama.com/windows), then download these models in PowerShell:

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

`qwen3.5:4b` is the chat model (about 3.3 to 4 GB on Ollama, depending on the tag); `qwen3-embedding:0.6b` turns document passages into searchable vectors (about 639 MB). The Qwen3.5 4B tag is listed under Apache 2.0 on Ollama, and the Qwen team says its Qwen3 embedding models are Apache 2.0. Check the license for the exact downloaded tags before distributing a commercial product. Actual speed depends on your computer. Start with the 4B chat model; if it runs too slowly, consider a hosted model provider after reviewing privacy and cost.

## Run on Windows

Open PowerShell in this folder. Python 3.12 is supported.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://localhost:8501>. Choose **Load sample support FAQ** to try the demo, or upload your own approved support files and choose **Index uploaded documents**.

## Verification status

The workflow tests pass in this workspace and the Streamlit page starts locally. The project was tested with Python 3.13; Python 3.12 is supported by the declared library versions but was not available for a direct test here. Ollama and the model weights were not installed in this workspace, so the actual Qwen answer generation still needs to be tried after you download the two models. See [test results](docs/TEST_RESULTS.md).

To run the automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

If you are already inside a folder named `backend` from another project, do not use that project's commands here. This app has its own `app.py` and `requirements.txt` in the `CustomerSupportAgent` folder.

## Settings

Copy `.env.example` to `.env` and edit as needed:

- `BUSINESS_NAME`: label shown in the app.
- `OLLAMA_BASE_URL`: local Ollama service address.
- `OLLAMA_CHAT_MODEL`: model tag used for answers.
- `OLLAMA_EMBEDDING_MODEL`: model tag used for document search.
- `RETRIEVAL_SCORE_THRESHOLD`: minimum evidence relevance before asking the model to answer.
- `MAX_UPLOAD_MB`: upload size limit per file and for the combined upload.

## Important limits before selling

The starter is suitable for a local demo or a supervised pilot. Before hosting it for customers, add login and company isolation, a real help-desk handoff, retention/deletion controls, monitoring, abuse limits, and customer-specific quality testing. Read [commercial readiness](docs/COMMERCIAL_READINESS.md) before using real customer information.

## References

- [LangGraph Python quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangChain Ollama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [Qwen3.5 4B on Ollama](https://ollama.com/library/qwen3.5:4b)
- [Qwen3 Embedding 0.6B on Ollama](https://ollama.com/library/qwen3-embedding:0.6b)
- [Qwen3 Embedding model license note](https://qwenlm.github.io/blog/qwen3-embedding/)
- [Ollama Windows setup](https://docs.ollama.com/windows)

