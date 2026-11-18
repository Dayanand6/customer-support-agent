# Test results

Checked 17 November 2026 on Windows with Python 3.13.

| Check | Result |
|---|---|
| Python dependency consistency (`pip check`) | Passed; no broken requirements |
| Agent workflow and handoff tests | 4 passed |
| Document parsing, splitting, and filename tests | 4 passed |
| Streamlit startup | Passed; local health endpoint returned `ok` |
| Local Qwen model inference | Not run; Ollama and model weights were not installed in this environment |

The automated tests use a fake chat model and fake search results to check the LangGraph decisions deterministically. After installing Ollama and pulling the model tags in the README, ask the sample FAQ both an in-scope question (for example, "What is the return window?") and an unsupported question (for example, "Can you guarantee delivery tomorrow?"). The first should cite the sample policy; the second should be flagged for a human.

