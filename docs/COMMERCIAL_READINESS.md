# Commercial readiness

This is a working local pilot for customer-support questions grounded in a company's documents. It is not yet a hosted, multi-customer service and should not be advertised as one.

## What works in this starter

- Reads PDF, DOCX, Markdown, and plain-text support documents.
- Splits and embeds documents with a local Ollama embedding model.
- Uses LangChain and a LangGraph workflow to retrieve evidence, answer, or flag a human handoff.
- Shows which uploaded files and PDF pages were used.
- Avoids external agent tools, so the model cannot issue refunds, update orders, or make account changes.
- Keeps uploaded text and vectors in the browser session's server memory; it does not write customer documents to a database.

## Before charging customers

1. Choose the legal terms for the application and review each model, package, and runtime license. The suggested Qwen chat model is listed under Apache 2.0 and Qwen says the embedding series is Apache 2.0; preserve notices and check the license for the exact downloaded tag.
2. Add authentication, company/workspace separation, admin controls, and tested data-deletion behavior before hosting for multiple customers.
3. Decide how real human support handoffs will reach a support team (for example, a help-desk integration). The current app only flags a handoff and lets the operator download a transcript; it does not send a ticket or email.
4. Test answer quality and refusal behavior against each customer's approved documents, including outdated, conflicting, and prompt-injection-containing content.
5. Add monitoring, rate limits, request limits, backup/retention policy, and an incident process before exposing a hosted service to the internet.
6. Ask pilot customers to approve document use and data handling. Do not upload secrets, payment data, or personal records to the demo.

## Product claims

Describe this as a local document-based support assistant with citations and human handoff. Do not promise perfect accuracy or claim that it is production-ready, secure for multi-tenancy, or connected to customer systems.

