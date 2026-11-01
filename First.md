# # Project: FastAPI Backend for SLM-Based Document Analysis

## 1. Project Objective

I am developing an M.Tech project titled:

"A Study of Parameter-Efficient Fine-Tuning of Small Language Models (SLMs) for Document Analysis."

I need a Python backend that allows a frontend web application to upload PDF documents, extract and split their text into chunks, accept questions about those documents, and generate answers using a Small Language Model (SLM).

I already have a separate Google Colab notebook used for preparing document text for model training. That notebook is for the training-data preparation workflow.

The backend you create now is for document question answering during inference. Do not confuse these two workflows or modify the existing training notebook.

The backend should be a separate Python application that runs locally on my MacBook Air.

Use FastAPI to expose REST API endpoints that a React frontend can call later.

I am primarily a Python backend developer, so write clean, modular, well-documented Python code and explain how to configure and run the application.

Do not only provide code snippets. Create the actual project files if your environment supports file creation.

## 2. Required Technology Stack

Use:

- Python 3.11 or a compatible version supported by the selected model dependencies.
- FastAPI for the REST API.
- Uvicorn to run the API server.
- PyMuPDF or another suitable library for extracting text from PDFs.
- Pydantic for request and response validation.
- The Transformers library if required by the configured model.
- PEFT if required to load a saved LoRA adapter.
- PyTorch for model inference.
- A suitable lightweight retrieval method for selecting relevant document chunks.

Use additional libraries only when they are necessary.

Avoid introducing a database, vector database, Docker, distributed processing, authentication infrastructure, or unnecessary frameworks in the initial implementation.

## 3. Application Architecture

Organise the backend into separate modules with clear responsibilities.

Use a project structure similar to:

slm-document-analysis-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── document_processor.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── model_service.py
│   └── storage.py
├── tests/
│   ├── test_document_processor.py
│   ├── test_chunking.py
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

You may adjust the structure if there is a good technical reason, but keep the implementation modular and easy to understand.

## 4. Configuration

Create a configuration module that reads settings from environment variables.

Include configurable values for:

- Model identifier or local base-model path.
- Saved LoRA adapter path, if applicable.
- Model device and precision settings.
- Maximum upload size.
- Chunk size.
- Chunk overlap.
- Maximum number of retrieved chunks.
- Maximum context length.
- Maximum generated answer length.
- Any retrieval settings required by the implementation.

Do not invent the location of my saved model or adapter.

If the model identifier or adapter path is unknown, use clearly documented placeholders and explain exactly what I must configure.

Do not commit credentials, API keys, tunnel tokens, or other secrets.

## 5. PDF Upload and Text Extraction

Create a reusable document-processing function.

It should:

1. Accept an uploaded PDF.
2. Validate that the uploaded file is a supported PDF.
3. Validate that the file is not empty and does not exceed the configured size limit.
4. Extract text page by page.
5. Preserve page boundaries and page numbers where possible.
6. Handle malformed PDFs and extraction errors.
7. Detect PDFs from which no meaningful text can be extracted.
8. Return the extracted text and useful metadata.

Use an appropriate PDF text-extraction library.

For the initial version, support text-based PDFs.

Do not claim to support scanned PDFs unless OCR is actually implemented.

Do not train or fine-tune the model when a document is uploaded.

The uploaded document is only being processed for inference.

## 6. Text Chunking

Implement a reusable chunking function.

After text extraction, split the document into manageable, overlapping chunks.

Requirements:

- Make chunk size and overlap configurable.
- Preserve the document's original text.
- Avoid producing empty chunks.
- Preserve page metadata where possible.
- Assign unique IDs to chunks.
- Associate every chunk with its document ID.
- Handle documents of different lengths.
- Avoid unnecessarily splitting text in the middle of words.
- Prevent the overlap from producing an infinite loop or duplicate-only chunks.
- Keep chunk sizes compatible with the chosen model's context limitations.

Use a straightforward implementation that is easy to explain during an academic presentation.

Do not introduce complicated semantic chunking unless necessary.

## 7. Temporary Document Storage

Create a storage module that keeps processed documents and their chunks in memory.

Each processed document should have a unique document ID.

Store the following information where available:

- Document ID.
- Original filename.
- Extracted text or a reference to it.
- Document metadata.
- Text chunks.
- Chunk IDs.
- Page metadata.
- Processing status.

The question-answering endpoint must use the document ID to locate the correct document.

A question about one document must never retrieve chunks from a different document.

Support multiple documents in the current backend process.

Implement a cleanup mechanism for old documents if practical.

Explain that in-memory storage is temporary, is not shared across separate backend processes, and is lost when the backend restarts.

Do not claim that uploaded files or document content are stored permanently.

Do not save uploaded documents to arbitrary filesystem paths supplied by the client.

## 8. Document Retrieval

Implement a retrieval module that selects relevant chunks for a user's question.

For the initial proof of concept, use a practical and lightweight retrieval strategy.

Consider either:

- A simple lexical retrieval approach that does not require an additional embedding model; or
- Embedding-based semantic retrieval with cosine similarity, if it is feasible with the available environment and model requirements.

Choose one approach and explain the reasoning.

Keep the retrieval interface modular so it can be upgraded later.

Requirements:

- Retrieve chunks only from the specified document.
- Rank chunks by relevance to the question.
- Limit the number and total length of selected chunks.
- Preserve the source chunk IDs and page numbers.
- Handle cases where no suitable chunks are found.
- Avoid passing the entire document to the model unnecessarily.

If using embedding-based retrieval, do not claim that semantic retrieval is implemented unless the code actually generates embeddings and computes relevance scores.

Do not require a vector database for the initial version.

## 9. Model Loading and Inference

Create a separate model service responsible for loading the configured SLM and generating answers.

The implementation should support my existing base model and saved LoRA adapter if they are available.

If using a LoRA adapter:

1. Load the correct base model.
2. Load the corresponding adapter using PEFT.
3. Configure the model for inference.
4. Reuse the loaded model across requests.
5. Avoid reloading the model for every question.

Do not invent a base-model identifier or adapter path.

If configuration is missing, return a clear startup error or provide a documented configuration step. Do not silently replace my intended model with a different model.

Make device selection configurable.

Do not assume that CUDA is available on my MacBook Air. If the selected model supports CPU or Apple Silicon inference, document the relevant configuration and limitations.

If the model is too large or requires a GPU unavailable locally, clearly explain that the model inference component must run in a suitable remote environment, such as Google Colab, or on another supported machine.

Keep model loading separate from document extraction and retrieval.

### Inference function

Create a reusable function that accepts:

- The user's question.
- The retrieved document context.
- Relevant source metadata.

The function should:

1. Construct the model prompt.
2. Use the configured model's appropriate prompt template.
3. Generate an answer.
4. Return the answer and inference metadata.

Ensure that only newly generated tokens are decoded as the answer.

Handle generation errors and timeouts appropriately.

Do not fabricate answers or return hardcoded responses.

## 10. Prompt Construction

Create a dedicated prompt-construction function.

The prompt should include:

- Clear instructions to answer from the document context.
- The relevant retrieved text chunks.
- The user's question.

Use an instruction similar to:

"You are a document question-answering assistant. Answer the question using only the supplied document context. Do not invent facts that are not supported by the context. If the answer cannot be found in the supplied context, clearly state that you could not find the answer in the document."

Adapt this instruction to the actual model's chat template or instruction format.

Do not assume every model uses the same prompt format.

Keep the prompt length within the model's supported context window.

The application should not claim that its answers are guaranteed to be correct.

## 11. FastAPI Endpoints

Implement the following REST API endpoints.

### Endpoint A: Health Check

Method: GET

Path: /health

Purpose:
- Confirm that the backend is running.
- Report model readiness when it can be determined reliably.
- Return useful status information.

Example response:

{
  "status": "ok",
  "model_ready": true
}

The value of model_ready must reflect the actual model service state.

### Endpoint B: Upload and Process a Document

Method: POST

Path: /documents

Accept a PDF through multipart/form-data.

Purpose:
1. Validate the uploaded PDF.
2. Extract its text.
3. Split the text into chunks.
4. Store the document and chunks in memory.
5. Return the document ID and processing metadata.

Return information such as:

{
  "document_id": "generated-document-id",
  "filename": "example.pdf",
  "status": "processed",
  "page_count": 5,
  "character_count": 12000,
  "word_count": 2000,
  "chunk_count": 8,
  "text_preview": "A bounded preview of the actual extracted text."
}

This is an illustrative response schema. Populate every field from actual processing results.

Do not invent page counts, word counts, chunk counts or extracted text.

Return an appropriate HTTP error if extraction fails or no meaningful text can be extracted.

Do not return an unbounded amount of text in the response.

### Endpoint C: Ask a Question About a Document

Method: POST

Path: /documents/{document_id}/questions

Accept a JSON request such as:

{
  "question": "What is the main conclusion of this report?"
}

The backend should:

1. Validate the document ID.
2. Validate that the question is not empty.
3. Locate the specified processed document.
4. Retrieve the most relevant chunks from that document.
5. Construct the prompt.
6. Call the configured SLM.
7. Return the generated answer and supporting evidence, if available.

Return a response similar to:

{
  "document_id": "generated-document-id",
  "question": "What is the main conclusion of this report?",
  "answer": "The actual answer generated by the configured model.",
  "model_name": "Actual configured model name",
  "inference_time_seconds": 1.25,
  "supporting_passages": [
    {
      "text": "Actual text retrieved from the document.",
      "page_number": 3,
      "chunk_id": "actual-chunk-id"
    }
  ]
}

The values above are illustrative only.

Populate the answer, model metadata, timing and supporting passages using actual results.

If no reliable supporting passage is found, handle that situation explicitly.

Do not fabricate source text, page numbers, confidence scores or answers.

Return meaningful HTTP errors for unknown documents, invalid questions, model failures and other errors.

### Endpoint D: Delete a Document

Method: DELETE

Path: /documents/{document_id}

Purpose:
- Delete the specified processed document and its chunks from in-memory storage.
- Return a meaningful response.

Do not claim to delete permanent files if the implementation only removes in-memory data.

Return an appropriate error if the document does not exist.

## 12. API Validation and Error Handling

Implement consistent error responses.

Handle:
- Unsupported file types.
- Empty uploads.
- Oversized PDFs.
- Malformed PDFs.
- PDFs without extractable text.
- Invalid document IDs.
- Empty questions.
- Model loading failures.
- Inference failures.
- Retrieval failures.
- Unexpected backend exceptions.

Do not expose stack traces, secrets or internal filesystem paths in production-facing error responses.

Log useful diagnostic information without unnecessarily logging entire documents or sensitive content.

## 13. CORS and Frontend Integration

Configure FastAPI CORS for the local React frontend.

Make the allowed frontend origins configurable.

For local development, allow the actual Vite frontend origin, such as:

http://localhost:5173

Do not use unrestricted CORS in place of proper configuration.

The API must support the frontend's intended workflow:

1. Call GET /health.
2. Upload a PDF through POST /documents.
3. Receive the document ID and extraction metadata.
4. Ask questions through POST /documents/{document_id}/questions.
5. Display the returned answer and supporting evidence.
6. Ask additional questions without uploading the same PDF again.
7. Delete the current document using DELETE /documents/{document_id} when appropriate.

Document the actual request and response formats so a separate frontend can integrate with the backend without guessing.

## 14. Local Development and Remote Model Support

The default backend should run locally on my MacBook Air.

However, the model may require a GPU or more memory than is available locally.

Design the model service so that the inference implementation can be adapted to a remote model service if necessary.

Do not introduce remote infrastructure unless it is actually required.

If the model must run in Google Colab:
- Explain which part of the backend must run remotely.
- Explain how the local FastAPI backend can communicate with the remote inference service, if that architecture is chosen.
- Do not assume that Colab's localhost is directly accessible from my MacBook.
- Document the additional networking configuration required.
- Do not expose credentials or tunnel tokens in frontend code.

Do not create a fake remote endpoint or claim remote integration is complete unless it is implemented.

## 15. Testing

Create tests for the document-processing functions, chunking and API validation.

Use temporary test PDFs or generated test fixtures where practical.

Do not require downloading a large model just to run basic unit tests.

Separate unit tests from tests that require the actual model.

Test at least the following:

1. A valid text-based PDF can be processed.
2. Invalid file types are rejected.
3. Empty or unextractable documents are rejected.
4. Chunking produces valid, non-empty chunks.
5. Chunks retain their document IDs and page metadata.
6. A question can only retrieve chunks belonging to its specified document.
7. Unknown document IDs return appropriate errors.
8. Empty questions are rejected.
9. Model-not-ready errors are handled clearly.
10. Document deletion removes the associated in-memory data.

Clearly distinguish tests that have actually passed from tests that have only been written.

## 16. Deliverables

Generate:

1. The complete Python backend project.
2. app/main.py with the FastAPI application.
3. Separate document extraction, chunking, retrieval, storage and model-service modules.
4. requirements.txt.
5. .env.example.
6. .gitignore.
7. Unit tests.
8. README.md.
9. API documentation with example requests and responses.
10. Exact commands for creating a virtual environment and starting the application on macOS.
11. Instructions for configuring the base model and LoRA adapter.
12. A short example showing how a frontend can call each endpoint.

Use the actual generated project structure when writing the README.

Do not provide commands that refer to files that do not exist.

## 17. Implementation Instructions

Before coding, determine whether you have access to my existing model configuration, saved adapter and training notebook.

If those resources are available, inspect them and reuse the correct model details.

If they are not available:
- Do not invent their locations.
- Use clearly documented configuration placeholders.
- List the information I need to provide.

Prioritise a minimal, working backend over unnecessary features.

Do not add authentication, persistent storage, a vector database or OCR unless required for the first working version.

After generating the code, explain:

1. Which files were created.
2. How the API works.
3. Which configuration values I need to change.
4. How to start the backend.
5. How to test each endpoint.
6. Whether the model can run locally or needs remote inference.
7. Any functionality that remains incomplete.

---

**Status:** active  **Date:** 2026-10-30

---

### 👤 User

# Project: FastAPI Backend for SLM-Based Document Analysis

## 1. Project Objective

I am developing an M.Tech project titled:

"A Study of Parameter-Efficient Fine-Tuning of Small Language Models (SLMs) for Document Analysis."

I need a Python backend that allows a frontend web application to upload PDF documents, extract and split their text into chunks, accept questions about those documents, and generate answers using a Small Language Model (SLM).

I already have a separate Google Colab notebook used for preparing document text for model training. That notebook is for the training-data preparation workflow.

The backend you create now is for document question answering during inference. Do not confuse these two workflows or modify the existing training notebook.

The backend should be a separate Python application that runs locally on my MacBook Air.

Use FastAPI to expose REST API endpoints that a React frontend can call later.

I am primarily a Python backend developer, so write clean, modular, well-documented Python code and explain how to configure and run the application.

Do not only provide code snippets. Create the actual project files if your environment supports file creation.

## 2. Required Technology Stack

Use:

- Python 3.11 or a compatible version supported by the selected model dependencies.
- FastAPI for the REST API.
- Uvicorn to run the API server.
- PyMuPDF or another suitable library for extracting text from PDFs.
- Pydantic for request and response validation.
- The Transformers library if required by the configured model.
- PEFT if required to load a saved LoRA adapter.
- PyTorch for model inference.
- A suitable lightweight retrieval method for selecting relevant document chunks.

Use additional libraries only when they are necessary.

Avoid introducing a database, vector database, Docker, distributed processing, authentication infrastructure, or unnecessary frameworks in the initial implementation.

## 3. Application Architecture

Organise the backend into separate modules with clear responsibilities.

Use a project structure similar to:

slm-document-analysis-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── document_processor.py
│   ├── chunking.py
│   ├── retrieval.py
│   ├── model_service.py
│   └── storage.py
├── tests/
│   ├── test_document_processor.py
│   ├── test_chunking.py
│   └── test_api.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md

You may adjust the structure if there is a good technical reason, but keep the implementation modular and easy to understand.

## 4. Configuration

Create a configuration module that reads settings from environment variables.

Include configurable values for:

- Model identifier or local base-model path.
- Saved LoRA adapter path, if applicable.
- Model device and precision settings.
- Maximum upload size.
- Chunk size.
- Chunk overlap.
- Maximum number of retrieved chunks.
- Maximum context length.
- Maximum generated answer length.
- Any retrieval settings required by the implementation.

Do not invent the location of my saved model or adapter.

If the model identifier or adapter path is unknown, use clearly documented placeholders and explain exactly what I must configure.

Do not commit credentials, API keys, tunnel tokens, or other secrets.

## 5. PDF Upload and Text Extraction

Create a reusable document-processing function.

It should:

1. Accept an uploaded PDF.
2. Validate that the uploaded file is a supported PDF.
3. Validate that the file is not empty and does not exceed the configured size limit.
4. Extract text page by page.
5. Preserve page boundaries and page numbers where possible.
6. Handle malformed PDFs and extraction errors.
7. Detect PDFs from which no meaningful text can be extracted.
8. Return the extracted text and useful metadata.

Use an appropriate PDF text-extraction library.

For the initial version, support text-based PDFs.

Do not claim to support scanned PDFs unless OCR is actually implemented.

Do not train or fine-tune the model when a document is uploaded.

The uploaded document is only being processed for inference.

## 6. Text Chunking

Implement a reusable chunking function.

After text extraction, split the document into manageable, overlapping chunks.

Requirements:

- Make chunk size and overlap configurable.
- Preserve the document's original text.
- Avoid producing empty chunks.
- Preserve page metadata where possible.
- Assign unique IDs to chunks.
- Associate every chunk with its document ID.
- Handle documents of different lengths.
- Avoid unnecessarily splitting text in the middle of words.
- Prevent the overlap from producing an infinite loop or duplicate-only chunks.
- Keep chunk sizes compatible with the chosen model's context limitations.

Use a straightforward implementation that is easy to explain during an academic presentation.

Do not introduce complicated semantic chunking unless necessary.

## 7. Temporary Document Storage

Create a storage module that keeps processed documents and their chunks in memory.

Each processed document should have a unique document ID.

Store the following information where available:

- Document ID.
- Original filename.
- Extracted text or a reference to it.
- Document metadata.
- Text chunks.
- Chunk IDs.
- Page metadata.
- Processing status.

The question-answering endpoint must use the document ID to locate the correct document.

A question about one document must never retrieve chunks from a different document.

Support multiple documents in the current backend process.

Implement a cleanup mechanism for old documents if practical.

Explain that in-memory storage is temporary, is not shared across separate backend processes, and is lost when the backend restarts.

Do not claim that uploaded files or document content are stored permanently.

Do not save uploaded documents to arbitrary filesystem paths supplied by the client.

## 8. Document Retrieval

Implement a retrieval module that selects relevant chunks for a user's question.

For the initial proof of concept, use a practical and lightweight retrieval strategy.

Consider either:

- A simple lexical retrieval approach that does not require an additional embedding model; or
- Embedding-based semantic retrieval with cosine similarity, if it is feasible with the available environment and model requirements.

Choose one approach and explain the reasoning.

Keep the retrieval interface modular so it can be upgraded later.

Requirements:

- Retrieve chunks only from the specified document.
- Rank chunks by relevance to the question.
- Limit the number and total length of selected chunks.
- Preserve the source chunk IDs and page numbers.
- Handle cases where no suitable chunks are found.
- Avoid passing the entire document to the model unnecessarily.

If using embedding-based retrieval, do not claim that semantic retrieval is implemented unless the code actually generates embeddings and computes relevance scores.

Do not require a vector database for the initial version.

## 9. Model Loading and Inference

Create a separate model service responsible for loading the configured SLM and generating answers.

The implementation should support my existing base model and saved LoRA adapter if they are available.

If using a LoRA adapter:

1. Load the correct base model.
2. Load the corresponding adapter using PEFT.
3. Configure the model for inference.
4. Reuse the loaded model across requests.
5. Avoid reloading the model for every question.

Do not invent a base-model identifier or adapter path.

If configuration is missing, return a clear startup error or provide a documented configuration step. Do not silently replace my intended model with a different model.

Make device selection configurable.

Do not assume that CUDA is available on my MacBook Air. If the selected model supports CPU or Apple Silicon inference, document the relevant configuration and limitations.

If the model is too large or requires a GPU unavailable locally, clearly explain that the model inference component must run in a suitable remote environment, such as Google Colab, or on another supported machine.

Keep model loading separate from document extraction and retrieval.

### Inference function

Create a reusable function that accepts:

- The user's question.
- The retrieved document context.
- Relevant source metadata.

The function should:

1. Construct the model prompt.
2. Use the configured model's appropriate prompt template.
3. Generate an answer.
4. Return the answer and inference metadata.

Ensure that only newly generated tokens are decoded as the answer.

Handle generation errors and timeouts appropriately.

Do not fabricate answers or return hardcoded responses.

## 10. Prompt Construction

Create a dedicated prompt-construction function.

The prompt should include:

- Clear instructions to answer from the document context.
- The relevant retrieved text chunks.
- The user's question.

Use an instruction similar to:

"You are a document question-answering assistant. Answer the question using only the supplied document context. Do not invent facts that are not supported by the context. If the answer cannot be found in the supplied context, clearly state that you could not find the answer in the document."

Adapt this instruction to the actual model's chat template or instruction format.

Do not assume every model uses the same prompt format.

Keep the prompt length within the model's supported context window.

The application should not claim that its answers are guaranteed to be correct.

## 11. FastAPI Endpoints

Implement the following REST API endpoints.

### Endpoint A: Health Check

Method: GET

Path: /health

Purpose:
- Confirm that the backend is running.
- Report model readiness when it can be determined reliably.
- Return useful status information.

Example response:

{
  "status": "ok",
  "model_ready": true
}

The value of model_ready must reflect the actual model service state.

### Endpoint B: Upload and Process a Document

Method: POST

Path: /documents

Accept a PDF through multipart/form-data.

Purpose:
1. Validate the uploaded PDF.
2. Extract its text.
3. Split the text into chunks.
4. Store the document and chunks in memory.
5. Return the document ID and processing metadata.

Return information such as:

{
  "document_id": "generated-document-id",
  "filename": "example.pdf",
  "status": "processed",
  "page_count": 5,
  "character_count": 12000,
  "word_count": 2000,
  "chunk_count": 8,
  "text_preview": "A bounded preview of the actual extracted text."
}

This is an illustrative response schema. Populate every field from actual processing results.

Do not invent page counts, word counts, chunk counts or extracted text.

Return an appropriate HTTP error if extraction fails or no meaningful text can be extracted.

Do not return an unbounded amount of text in the response.

### Endpoint C: Ask a Question About a Document

Method: POST

Path: /documents/{document_id}/questions

Accept a JSON request such as:

{
  "question": "What is the main conclusion of this report?"
}

The backend should:

1. Validate the document ID.
2. Validate that the question is not empty.
3. Locate the specified processed document.
4. Retrieve the most relevant chunks from that document.
5. Construct the prompt.
6. Call the configured SLM.
7. Return the generated answer and supporting evidence, if available.

Return a response similar to:

{
  "document_id": "generated-document-id",
  "question": "What is the main conclusion of this report?",
  "answer": "The actual answer generated by the configured model.",
  "model_name": "Actual configured model name",
  "inference_time_seconds": 1.25,
  "supporting_passages": [
    {
      "text": "Actual text retrieved from the document.",
      "page_number": 3,
      "chunk_id": "actual-chunk-id"
    }
  ]
}

The values above are illustrative only.

Populate the answer, model metadata, timing and supporting passages using actual results.

If no reliable supporting passage is found, handle that situation explicitly.

Do not fabricate source text, page numbers, confidence scores or answers.

Return meaningful HTTP errors for unknown documents, invalid questions, model failures and other errors.

### Endpoint D: Delete a Document

Method: DELETE

Path: /documents/{document_id}

Purpose:
- Delete the specified processed document and its chunks from in-memory storage.
- Return a meaningful response.

Do not claim to delete permanent files if the implementation only removes in-memory data.

Return an appropriate error if the document does not exist.

## 12. API Validation and Error Handling

Implement consistent error responses.

Handle:
- Unsupported file types.
- Empty uploads.
- Oversized PDFs.
- Malformed PDFs.
- PDFs without extractable text.
- Invalid document IDs.
- Empty questions.
- Model loading failures.
- Inference failures.
- Retrieval failures.
- Unexpected backend exceptions.

Do not expose stack traces, secrets or internal filesystem paths in production-facing error responses.

Log useful diagnostic information without unnecessarily logging entire documents or sensitive content.

## 13. CORS and Frontend Integration

Configure FastAPI CORS for the local React frontend.

Make the allowed frontend origins configurable.

For local development, allow the actual Vite frontend origin, such as:

http://localhost:5173

Do not use unrestricted CORS in place of proper configuration.

The API must support the frontend's intended workflow:

1. Call GET /health.
2. Upload a PDF through POST /documents.
3. Receive the document ID and extraction metadata.
4. Ask questions through POST /documents/{document_id}/questions.
5. Display the returned answer and supporting evidence.
6. Ask additional questions without uploading the same PDF again.
7. Delete the current document using DELETE /documents/{document_id} when appropriate.

Document the actual request and response formats so a separate frontend can integrate with the backend without guessing.

## 14. Local Development and Remote Model Support

The default backend should run locally on my MacBook Air.

However, the model may require a GPU or more memory than is available locally.

Design the model service so that the inference implementation can be adapted to a remote model service if necessary.

Do not introduce remote infrastructure unless it is actually required.

If the model must run in Google Colab:
- Explain which part of the backend must run remotely.
- Explain how the local FastAPI backend can communicate with the remote inference service, if that architecture is chosen.
- Do not assume that Colab's localhost is directly accessible from my MacBook.
- Document the additional networking configuration required.
- Do not expose credentials or tunnel tokens in frontend code.

Do not create a fake remote endpoint or claim remote integration is complete unless it is implemented.

## 15. Testing

Create tests for the document-processing functions, chunking and API validation.

Use temporary test PDFs or generated test fixtures where practical.

Do not require downloading a large model just to run basic unit tests.

Separate unit tests from tests that require the actual model.

Test at least the following:

1. A valid text-based PDF can be processed.
2. Invalid file types are rejected.
3. Empty or unextractable documents are rejected.
4. Chunking produces valid, non-empty chunks.
5. Chunks retain their document IDs and page metadata.
6. A question can only retrieve chunks belonging to its specified document.
7. Unknown document IDs return appropriate errors.
8. Empty questions are rejected.
9. Model-not-ready errors are handled clearly.
10. Document deletion removes the associated in-memory data.

Clearly distinguish tests that have actually passed from tests that have only been written.

## 16. Deliverables

Generate:

1. The complete Python backend project.
2. app/main.py with the FastAPI application.
3. Separate document extraction, chunking, retrieval, storage and model-service modules.
4. requirements.txt.
5. .env.example.
6. .gitignore.
7. Unit tests.
8. README.md.
9. API documentation with example requests and responses.
10. Exact commands for creating a virtual environment and starting the application on macOS.
11. Instructions for configuring the base model and LoRA adapter.
12. A short example showing how a frontend can call each endpoint.

Use the actual generated project structure when writing the README.

Do not provide commands that refer to files that do not exist.

## 17. Implementation Instructions

Before coding, determine whether you have access to my existing model configuration, saved adapter and training notebook.

If those resources are available, inspect them and reuse the correct model details.

If they are not available:
- Do not invent their locations.
- Use clearly documented configuration placeholders.
- List the information I need to provide.

Prioritise a minimal, working backend over unnecessary features.

Do not add authentication, persistent storage, a vector database or OCR unless required for the first working version.

After generating the code, explain:

1. Which files were created.
2. How the API works.
3. Which configuration values I need to change.
4. How to start the backend.
5. How to test each endpoint.
6. Whether the model can run locally or needs remote inference.
7. Any functionality that remains incomplete.

---

### 🤖 Assistant

> 🔧 called: list_files

I'll start by checking what already exists in the workspace (model config, adapter, notebook).

> 🔧 **list_files** — .

### 🤖 Assistant

> 🔧 called: execute_command, grep



> 🔧 **execute_command** — Get-ChildItem -Force . | Select-Object Name; Get-ChildItem -Force .. | Select-Object Name; wsl.exe --version 2>&1 | Select-Object -First 3

> 🔧 **grep** — adapter|LoRA|peft

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — Get-ChildItem -Force . ; Write-Output ---; Get-ChildItem -Force ../Mtech -Recurse -Depth 2 | Select-Object FullName

### 🤖 Assistant

> 🔧 called: read_file, read_file, read_file, read_file

The `Project` directory is empty, but the sibling `Mtech` folder looks like your training project. I'll read its model configuration so I reuse the real model details.

> 🔧 **read_file** — ../Mtech/src/data/prompts.py

> 🔧 **read_file** — ../Mtech/src/training/model_loader.py

> 🔧 **read_file** — ../Mtech/configs/pilot_squad_qa.yaml

> 🔧 **read_file** — ../Mtech/docs/progress.md

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "which python3 python3.11 python3.12; python3 --version; python3 -m venv --help >/dev/null 2>&1 && echo venv-ok; pip3 --version"

### 🤖 Assistant

> 🔧 called: update_todo_list, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file



> 🔧 **update_todo_list** — [-] Create core modules (config, errors, schemas, document_processor, chunking, storage, retrieval, prompting, model_service)
[ ] Create main.py FastAPI app
[ ] Create tests + pytest.ini
[ ] Create requirements, .env.example, .gitignore, README
[ ] Create venv in WSL, install deps, run tests
[ ] Final summary

> 🔧 **write_file** — slm-document-analysis-backend/app/__init__.py

> 🔧 **write_file** — slm-document-analysis-backend/app/config.py

> 🔧 **write_file** — slm-document-analysis-backend/app/errors.py

> 🔧 **write_file** — slm-document-analysis-backend/app/document_processor.py

> 🔧 **write_file** — slm-document-analysis-backend/app/chunking.py

> 🔧 **write_file** — slm-document-analysis-backend/app/retrieval.py

> 🔧 **write_file** — slm-document-analysis-backend/app/storage.py

> 🔧 **write_file** — slm-document-analysis-backend/app/prompting.py

### 🤖 Assistant

> 🔧 called: write_file, write_file



> 🔧 **write_file** — """Pydantic request/response models (the public API contract)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(examples=["ok"])
    model_ready: bool
    model_name: Optional[str] = None
    adapter_loaded: bool = False
    device: Optional[str] = None
    model_error: Optional[str] = None
    documents_in_memory: int = 0


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    page_count: int
    character_count: int
    word_count: int
    chunk_count: int
    empty_page_numbers: list[int]
    text_preview: str
    text_preview_truncated: bool


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class SupportingPassage(BaseModel):
    text: str
    page_number: int
    chunk_id: str
    retrieval_score: float = Field(description="BM25 ranking score; NOT a probability or confidence.")


class QuestionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    document_id: str
    question: str
    answer: Optional[str] = Field(
        description="Model-generated answer, or null when no relevant passage was found."
    )
    answer_generated: bool
    message: Optional[str] = None
    model_name: Optional[str] = None
    inference_time_seconds: Optional[float] = None
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    supporting_passages: list[SupportingPassage]
    disclaimer: str = (
        "Answers are generated by a small language model and may be incorrect. "
        "Verify them against the supporting passages."
    )


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


> 🔧 **write_file** — """Model service: loads the configured SLM (+ optional LoRA adapter) once and answers questions.

Kept independent of PDF extraction, chunking and retrieval. The rest of the app only
depends on the small ``AnswerGenerator`` protocol below, so a remote inference
implementation (e.g. an HTTP client for a model hosted on Colab) could replace
``ModelService`` without touching other modules. No remote implementation exists yet.

Heavy libraries (torch / transformers / peft) are imported lazily inside ``load`` so
the unit tests and the API can start without them being initialised.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from app.config import Settings
from app.prompting import build_user_message
from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


# --- Errors -------------------------------------------------------------------
class ModelServiceError(Exception):
    code = "model_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotConfiguredError(ModelServiceError):
    code = "model_not_configured"


class ModelNotReadyError(ModelServiceError):
    code = "model_not_ready"


class ModelLoadError(ModelServiceError):
    code = "model_load_failed"


class InferenceError(ModelServiceError):
    code = "inference_failed"


class InferenceTimeoutError(ModelServiceError):
    code = "inference_timeout"


# --- Public result / interface --------------------------------------------------
@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model_name: str
    prompt_tokens: int
    generated_tokens: int
    passages_used: int          # how many retrieved passages fit into the prompt
    device: str
    dtype: str


class AnswerGenerator(Protocol):
    """Interface the API depends on."""

    @property
    def is_ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    @property
    def status(self) -> dict[str, Any]: ...

    def generate_answer(
        self, question: str, passages: Sequence[RetrievedChunk]
    ) -> GenerationResult: ...


# --- Implementation -----------------------------------------------------------
class ModelService:
    """Local Transformers (+ PEFT) inference. The model is loaded once and reused."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "unloaded"
        self._dtype = "unloaded"
        self._adapter_loaded = False
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()  # one generation at a time

    # -- state -------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        base = self._settings.model_name_or_path or "unconfigured"
        # Local paths are reduced to their last component to avoid leaking filesystem layout.
        if os.path.isabs(base) or base.startswith(("~", ".")):
            base = os.path.basename(os.path.normpath(os.path.expanduser(base)))
        if self._adapter_loaded and self._settings.adapter_path:
            adapter = os.path.basename(os.path.normpath(self._settings.adapter_path))
            return f"{base} + LoRA adapter '{adapter}'"
        return base

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.is_ready,
            "model_name": self.model_name if self._settings.model_name_or_path else None,
            "adapter_loaded": self._adapter_loaded,
            "device": self._device,
            "load_error": self._load_error,
        }

    # -- loading -----------------------------------------------------------------
    def load(self) -> None:
        """Load tokenizer, base model and (optionally) the LoRA adapter.

        Raises ``ModelNotConfiguredError`` or ``ModelLoadError``. Safe to call again;
        it does nothing if the model is already loaded.
        """
        with self._load_lock:
            if self.is_ready:
                return
            s = self._settings
            if not s.model_name_or_path:
                msg = (
                    "MODEL_NAME_OR_PATH is not set. Set it (and ADAPTER_PATH if you use a "
                    "LoRA adapter) in the .env file."
                )
                self._load_error = msg
                raise ModelNotConfiguredError(msg)
            if s.adapter_path and not os.path.isdir(os.path.expanduser(s.adapter_path)):
                msg = "ADAPTER_PATH does not point to an existing directory."
                self._load_error = msg
                raise ModelLoadError(msg)

            try:
                self._load_unchecked()
                self._load_error = None
            except ModelServiceError:
                raise
            except Exception as exc:
                logger.exception("Model loading failed")
                self._model = None
                self._tokenizer = None
                self._load_error = f"Model loading failed ({type(exc).__name__}). See server logs."
                raise ModelLoadError(self._load_error) from exc

    def _load_unchecked(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self._settings
        device = self._resolve_device(torch)
        dtype_name, dtype = self._resolve_dtype(torch, device)
        logger.info("Loading model on device=%s dtype=%s", device, dtype_name)

        tokenizer = AutoTokenizer.from_pretrained(
            s.tokenizer_name_or_path or s.model_name_or_path,
            trust_remote_code=s.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            s.model_name_or_path,
            dtype=dtype,
            trust_remote_code=s.trust_remote_code,
        )

        adapter_loaded = False
        if s.adapter_path:
            from peft import PeftModel

            logger.info("Loading LoRA adapter")
            model = PeftModel.from_pretrained(model, os.path.expanduser(s.adapter_path))
            if s.merge_adapter:
                model = model.merge_and_unload()
            adapter_loaded = True

        model.to(device)
        model.eval()

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype = dtype_name
        self._adapter_loaded = adapter_loaded
        logger.info("Model ready")

    def _resolve_device(self, torch: Any) -> str:
        requested = self._settings.model_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("MODEL_DEVICE=cuda but CUDA is not available on this machine.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ModelLoadError("MODEL_DEVICE=mps but Apple MPS is not available on this machine.")
        return requested

    def _resolve_dtype(self, torch: Any, device: str) -> tuple[str, Any]:
        requested = self._settings.model_dtype
        if requested == "auto":
            requested = "float32" if device == "cpu" else "float16"
        return requested, getattr(torch, requested)

    # -- inference -----------------------------------------------------------------
    def generate_answer(
        self, question: str, passages: Sequence[RetrievedChunk]
    ) -> GenerationResult:
        """Build the prompt, generate, and decode ONLY the newly generated tokens."""
        if not self.is_ready:
            raise ModelNotReadyError(
                self._load_error or "The model is not loaded yet. Check /health for details."
            )
        import torch

        s = self._settings
        try:
            input_ids, used = self._build_inputs(question, passages)
        except Exception as exc:
            logger.exception("Prompt construction failed")
            raise InferenceError("Failed to build the model prompt.") from exc
        prompt_tokens = int(input_ids.shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": s.max_new_tokens,
            "max_time": s.generation_timeout_seconds,
            "do_sample": s.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if s.do_sample:
            gen_kwargs["temperature"] = s.temperature

        started = time.monotonic()
        try:
            with self._generate_lock, torch.inference_mode():
                attention_mask = torch.ones_like(input_ids)
                output = self._model.generate(
                    input_ids=input_ids.to(self._device),
                    attention_mask=attention_mask.to(self._device),
                    **gen_kwargs,
                )
        except Exception as exc:
            logger.exception("Generation failed")
            raise InferenceError(f"Generation failed ({type(exc).__name__}).") from exc
        elapsed = time.monotonic() - started

        new_tokens = output[0][prompt_tokens:]
        generated_tokens = int(new_tokens.shape[-1])
        # max_time stops generation early; treat an unfinished, timed-out run as a timeout.
        hit_eos = generated_tokens > 0 and int(new_tokens[-1]) == self._tokenizer.eos_token_id
        if elapsed >= s.generation_timeout_seconds and not hit_eos and generated_tokens < s.max_new_tokens:
            raise InferenceTimeoutError(
                f"Answer generation exceeded {s.generation_timeout_seconds:.0f} seconds."
            )

        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not answer:
            raise InferenceError("The model produced an empty answer.")

        return GenerationResult(
            answer=answer,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            passages_used=used,
            device=self._device,
            dtype=self._dtype,
        )

    def _build_inputs(
        self, question: str, passages: Sequence[RetrievedChunk]
    ) -> tuple[Any, int]:
        """Tokenize a prompt that fits the context budget.

        Lowest-ranked passages are dropped first; if a single passage is still too
        long it is truncated. Uses the tokenizer's chat template when it has one.
        """
        s = self._settings
        context_limit = getattr(self._tokenizer, "model_max_length", None)
        if not isinstance(context_limit, int) or context_limit > 10_000_000:
            context_limit = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)
        budget = s.max_input_tokens
        if isinstance(context_limit, int) and 0 < context_limit <= 10_000_000:
            budget = min(budget, context_limit - s.max_new_tokens)
        if budget < 64:
            raise ValueError("Token budget is too small; lower MAX_NEW_TOKENS.")

        selected = list(passages)
        while True:
            ids = self._encode(question, selected)
            if ids.shape[-1] <= budget or not selected:
                break
            if len(selected) > 1:
                selected.pop()
                continue
            # One passage left: shrink its text proportionally and retry.
            only = selected[0]
            ratio = max(0.1, (budget - 64) / ids.shape[-1])
            new_len = max(50, int(len(only.chunk.text) * ratio))
            if new_len >= len(only.chunk.text):
                break
            from dataclasses import replace

            selected = [replace(only, chunk=replace(only.chunk, text=only.chunk.text[:new_len]))]
        if ids.shape[-1] > budget:
            raise ValueError("Prompt does not fit within the model context window.")
        return ids, len(selected)

    def _encode(self, question: str, passages: Sequence[RetrievedChunk]) -> Any:
        user_message = build_user_message(question, passages)
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(
                [{"role": "user", "content": user_message}],
                tokenize=False,
                add_generation_prompt=True,
            )
            # The template already inserts special tokens such as BOS.
            return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        text = f"{user_message}\n\nAnswer:"
        return tok(text, return_tensors="pt")["input_ids"]


### 🤖 Assistant

> 🔧 called: write_file, write_file, write_file, write_file, write_file



> 🔧 **write_file** — # Secrets / local configuration
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/

# Models and uploads must never be committed
models/
adapters/
uploads/
*.safetensors
*.bin
*.pt
*.gguf

# OS / editors
.DS_Store
.idea/
.vscode/


> 🔧 **write_file** — fastapi>=0.115
uvicorn>=0.30
python-multipart>=0.0.9
pydantic>=2.7
pydantic-settings>=2.3
pymupdf>=1.24
torch>=2.4
transformers>=4.56
peft>=0.13
accelerate>=1.0
pytest>=8.0
httpx>=0.27


> 🔧 **write_file** — # Copy to ".env" and edit. Never commit ".env".

# ===== REQUIRED: your model (no default is assumed) =====
# Base model used when training the LoRA adapter: a Hugging Face id or a local directory.
MODEL_NAME_OR_PATH=
# Directory of your saved LoRA adapter (contains adapter_config.json). Leave empty to
# run the base model only.
ADAPTER_PATH=
# Optional: tokenizer location if different from the base model.
TOKENIZER_NAME_OR_PATH=

# ===== Device / precision =====
# auto | cpu | mps | cuda   (auto = cuda, else mps, else cpu)
MODEL_DEVICE=auto
# auto | float32 | float16 | bfloat16   (auto = float32 on cpu, float16 otherwise)
MODEL_DTYPE=auto
MERGE_ADAPTER=false
TRUST_REMOTE_CODE=false
LOAD_MODEL_ON_STARTUP=true

# ===== Generation =====
MAX_NEW_TOKENS=256
MAX_INPUT_TOKENS=3072
GENERATION_TIMEOUT_SECONDS=120
DO_SAMPLE=false
TEMPERATURE=0.7

# ===== Upload and chunking (sizes in characters) =====
MAX_UPLOAD_MB=20
MIN_TEXT_CHARS=20
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TEXT_PREVIEW_CHARS=500

# ===== Retrieval =====
RETRIEVAL_TOP_K=4
MAX_CONTEXT_CHARS=6000
BM25_K1=1.5
BM25_B=0.75

# ===== In-memory storage =====
DOCUMENT_TTL_MINUTES=60
MAX_DOCUMENTS=50

# ===== API =====
# Comma-separated browser origins allowed to call the API.
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
LOG_LEVEL=INFO


> 🔧 **write_file** — """FastAPI application: REST API for PDF upload and document question answering.

Run with:  ``uvicorn app.main:app --host 127.0.0.1 --port 8000``
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.chunking import chunk_pages
from app.config import Settings, get_settings
from app.document_processor import (
    DocumentProcessingError,
    EmptyFileError,
    FileTooLargeError,
    MalformedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    process_pdf,
    sanitize_filename,
)
from app.errors import ApiError
from app.model_service import (
    AnswerGenerator,
    InferenceTimeoutError,
    ModelNotConfiguredError,
    ModelNotReadyError,
    ModelService,
    ModelServiceError,
)
from app.retrieval import BM25Retriever
from app.schemas import (
    DeleteResponse,
    DocumentUploadResponse,
    ErrorResponse,
    HealthResponse,
    QuestionRequest,
    QuestionResponse,
    SupportingPassage,
)
from app.storage import DocumentStore, StoredDocument, new_document_id

logger = logging.getLogger("app")

_DOCUMENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")

_UPLOAD_STATUS = {
    UnsupportedFileTypeError: 415,
    EmptyFileError: 400,
    FileTooLargeError: 413,
    MalformedPDFError: 422,
    NoExtractableTextError: 422,
}

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    404: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def create_app(
    settings: Optional[Settings] = None,
    model_service: Optional[AnswerGenerator] = None,
) -> FastAPI:
    """Application factory. ``settings`` and ``model_service`` can be injected (used by tests)."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service: AnswerGenerator = model_service or ModelService(settings)
    store = DocumentStore(
        ttl_seconds=settings.document_ttl_minutes * 60,
        max_documents=settings.max_documents,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Only the built-in ModelService is loaded here; injected services manage themselves.
        if isinstance(service, ModelService) and settings.load_model_on_startup:
            try:
                await run_in_threadpool(service.load)
            except ModelServiceError as exc:
                # The API still starts so /health can report the problem clearly.
                logger.error("Model not available at startup: %s", exc.message)
        yield

    app = FastAPI(
        title="SLM Document Analysis API",
        description="Upload PDFs and ask questions answered by a small language model. "
        "Documents are held temporarily in memory only.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.model_service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    # --- error handlers ------------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Report only field location and a short message; never echo the submitted input.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}"
            for e in exc.errors()
        )
        return _error_response(422, "validation_error", f"Invalid request: {problems}")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error", exc_info=exc)
        return _error_response(500, "internal_error", "An unexpected error occurred.")

    # --- helpers ---------------------------------------------------------------
    def get_document_or_error(document_id: str) -> StoredDocument:
        if not _DOCUMENT_ID_RE.fullmatch(document_id):
            raise ApiError(400, "invalid_document_id", "The document ID is not valid.")
        doc = store.get(document_id)
        if doc is None:
            raise ApiError(
                404,
                "document_not_found",
                "Document not found. It may have been deleted or expired, or the "
                "backend was restarted (documents are kept in memory only).",
            )
        return doc

    # --- endpoints -------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        status = service.status
        return HealthResponse(
            status="ok",
            model_ready=bool(status.get("model_ready")),
            model_name=status.get("model_name"),
            adapter_loaded=bool(status.get("adapter_loaded")),
            device=status.get("device"),
            model_error=status.get("load_error"),
            documents_in_memory=store.count(),
        )

    @app.post(
        "/documents",
        response_model=DocumentUploadResponse,
        status_code=201,
        responses={**_ERROR_RESPONSES, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse}},
        tags=["documents"],
    )
    async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
        # Read at most max_bytes + 1 so oversized uploads are detected without buffering them all.
        data = await file.read(settings.max_upload_bytes + 1)
        filename = sanitize_filename(file.filename)
        try:
            extracted = await run_in_threadpool(
                process_pdf,
                file.filename,
                data,
                max_bytes=settings.max_upload_bytes,
                min_text_chars=settings.min_text_chars,
            )
        except DocumentProcessingError as exc:
            status = next(
                (s for cls, s in _UPLOAD_STATUS.items() if isinstance(exc, cls)), 422
            )
            logger.info("Upload rejected: %s", exc.code)
            raise ApiError(status, exc.code, exc.message) from exc
        except Exception as exc:
            logger.exception("Unexpected failure while processing PDF")
            raise ApiError(500, "processing_failed", "The document could not be processed.") from exc

        document_id = new_document_id()
        chunks = chunk_pages(
            extracted.pages, document_id, settings.chunk_size, settings.chunk_overlap
        )
        if not chunks:  # cannot normally happen after the meaningful-text check
            raise ApiError(422, "no_extractable_text", "No text chunks could be created.")
        doc = StoredDocument(
            document_id=document_id,
            filename=filename,
            pages=extracted.pages,
            chunks=chunks,
            retriever=BM25Retriever(chunks, k1=settings.bm25_k1, b=settings.bm25_b),
            page_count=extracted.page_count,
            character_count=extracted.character_count,
            word_count=extracted.word_count,
            empty_page_numbers=extracted.empty_page_numbers,
        )
        store.add(doc)
        logger.info(
            "Document processed id=%s pages=%d chunks=%d chars=%d",
            document_id, doc.page_count, len(chunks), doc.character_count,
        )

        preview_source = " ".join(doc.text_preview_source.split())
        limit = settings.text_preview_chars
        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            status=doc.status,
            page_count=doc.page_count,
            character_count=doc.character_count,
            word_count=doc.word_count,
            chunk_count=len(chunks),
            empty_page_numbers=doc.empty_page_numbers,
            text_preview=preview_source[:limit],
            text_preview_truncated=len(preview_source) > limit,
        )

    @app.post(
        "/documents/{document_id}/questions",
        response_model=QuestionResponse,
        responses=_ERROR_RESPONSES,
        tags=["documents"],
    )
    def ask_question(document_id: str, request: QuestionRequest) -> QuestionResponse:
        # Plain ``def``: FastAPI runs it in a worker thread, so slow generation does not
        # block the event loop.
        doc = get_document_or_error(document_id)

        if not service.is_ready:
            status = service.status
            raise ApiError(
                503,
                "model_not_ready",
                status.get("load_error")
                or "The language model is not ready. Check GET /health for details.",
            )

        try:
            retrieved = doc.retriever.retrieve(
                request.question,
                top_k=settings.retrieval_top_k,
                max_chars=settings.max_context_chars,
            )
        except Exception as exc:
            logger.exception("Retrieval failed")
            raise ApiError(500, "retrieval_failed", "Failed to retrieve document passages.") from exc

        if not retrieved:
            return QuestionResponse(
                document_id=document_id,
                question=request.question,
                answer=None,
                answer_generated=False,
                message="No passage in this document matched the question, so no answer "
                "was generated. Try rephrasing using words that appear in the document.",
                supporting_passages=[],
            )

        try:
            started = time.monotonic()
            result = service.generate_answer(request.question, retrieved)
            elapsed = time.monotonic() - started
        except InferenceTimeoutError as exc:
            raise ApiError(504, exc.code, exc.message) from exc
        except (ModelNotReadyError, ModelNotConfiguredError) as exc:
            raise ApiError(503, exc.code, exc.message) from exc
        except ModelServiceError as exc:
            raise ApiError(500, exc.code, exc.message) from exc

        passages = [
            SupportingPassage(
                text=r.chunk.text,
                page_number=r.chunk.page_number,
                chunk_id=r.chunk.chunk_id,
                retrieval_score=round(r.score, 4),
            )
            for r in retrieved[: result.passages_used]
        ]
        return QuestionResponse(
            document_id=document_id,
            question=request.question,
            answer=result.answer,
            answer_generated=True,
            model_name=result.model_name,
            inference_time_seconds=round(elapsed, 3),
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            supporting_passages=passages,
        )

    @app.delete(
        "/documents/{document_id}",
        response_model=DeleteResponse,
        responses=_ERROR_RESPONSES,
        tags=["documents"],
    )
    def delete_document(document_id: str) -> DeleteResponse:
        if not _DOCUMENT_ID_RE.fullmatch(document_id):
            raise ApiError(400, "invalid_document_id", "The document ID is not valid.")
        if not store.delete(document_id):
            raise ApiError(404, "document_not_found", "Document not found.")
        return DeleteResponse(
            document_id=document_id,
            deleted=True,
            message="The document and its chunks were removed from in-memory storage.",
        )

    return app


app = create_app()


> 🔧 **write_file** — [pytest]
testpaths = tests
pythonpath = .
markers =
    requires_model: needs a real configured model (MODEL_NAME_OR_PATH); skipped by default


### 🤖 Assistant

> 🔧 called: write_file, write_file, write_file, write_file, write_file, write_file

Writing `.gitignore` through the file tool is blocked, so I'll create it with a shell command later. Next are the tests, which use a fake model service so no real model is needed.

> 🔧 **write_file** — """Shared test fixtures: generated PDFs and a fake model service (no real model needed)."""

from __future__ import annotations

from typing import Any, Sequence

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_service import GenerationResult, ModelNotReadyError
from app.retrieval import RetrievedChunk


def make_pdf(pages: list[str]) -> bytes:
    """Build a text-based PDF where each string becomes one page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


class FakeModelService:
    """Stands in for ModelService. Records the passages it was given."""

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.ready,
            "model_name": "fake-model",
            "adapter_loaded": False,
            "device": "cpu",
            "load_error": None if self.ready else "Model has not been loaded.",
        }

    def generate_answer(
        self, question: str, passages: Sequence[RetrievedChunk]
    ) -> GenerationResult:
        if not self.ready:
            raise ModelNotReadyError("not ready")
        self.calls.append((question, list(passages)))
        return GenerationResult(
            answer="fake answer",
            model_name="fake-model",
            prompt_tokens=10,
            generated_tokens=2,
            passages_used=len(passages),
            device="cpu",
            dtype="float32",
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        max_upload_mb=1,
        chunk_size=200,
        chunk_overlap=40,
        retrieval_top_k=3,
    )


@pytest.fixture
def fake_model() -> FakeModelService:
    return FakeModelService()


@pytest.fixture
def client(settings: Settings, fake_model: FakeModelService) -> TestClient:
    return TestClient(create_app(settings=settings, model_service=fake_model))


> 🔧 **write_file** — 

> 🔧 **write_file** — import pytest

from app.chunking import chunk_pages, split_text
from app.document_processor import PageText
from app.retrieval import BM25Retriever
from app.storage import DocumentStore, StoredDocument

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 60).strip()


def test_split_text_spans_are_nonempty_and_within_size():
    spans = split_text(WORDS, 200, 40)
    assert len(spans) > 1
    for start, end in spans:
        assert 0 <= start < end <= len(WORDS)
        assert WORDS[start:end].strip() == WORDS[start:end]
        assert end - start <= 200


def test_split_text_does_not_cut_words():
    spans = split_text(WORDS, 200, 40)
    vocabulary = set(WORDS.split())
    for start, end in spans:
        assert set(WORDS[start:end].split()) <= vocabulary


def test_split_text_covers_all_text_and_overlaps():
    spans = split_text(WORDS, 200, 40)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(WORDS)
    for (_, prev_end), (next_start, next_end) in zip(spans, spans[1:]):
        assert next_start <= prev_end          # no gap: overlap present
        assert next_end > prev_end             # progress: no duplicate-only chunk


def test_split_text_terminates_on_unbroken_text():
    text = "x" * 1000
    spans = split_text(text, 100, 99)
    assert spans[-1][1] == 1000
    assert len(spans) < 1000


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_blank_text_produces_no_chunks(text):
    assert split_text(text, 100, 10) == []


def test_short_text_produces_single_chunk():
    assert split_text("short text", 100, 10) == [(0, 10)]


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)
    with pytest.raises(ValueError):
        split_text("abc", 1, 0)


def test_chunks_keep_document_id_page_and_unique_ids():
    pages = [PageText(1, WORDS), PageText(2, "A second page with different words."), PageText(3, "")]
    chunks = chunk_pages(pages, "doc1", 200, 40)
    assert chunks
    assert {c.document_id for c in chunks} == {"doc1"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.text.strip() for c in chunks)
    assert {c.page_number for c in chunks} == {1, 2}  # blank page 3 yields no chunks
    for c in chunks:
        page_text = pages[c.page_number - 1].text
        assert page_text[c.start_char:c.end_char] == c.text  # original text preserved


def _stored(doc_id: str, text: str) -> StoredDocument:
    chunks = chunk_pages([PageText(1, text)], doc_id, 200, 40)
    return StoredDocument(
        document_id=doc_id, filename=f"{doc_id}.pdf", pages=[PageText(1, text)], chunks=chunks,
        retriever=BM25Retriever(chunks), page_count=1, character_count=len(text), word_count=len(text.split()),
    )


def test_retrieval_only_returns_chunks_of_its_own_document():
    store = DocumentStore(ttl_seconds=3600, max_documents=10)
    store.add(_stored("docA", "The submarine dives deep into the ocean trench. " * 10))
    store.add(_stored("docB", "Quarterly revenue and profit margins improved. " * 10))
    hits = store.get("docA").retriever.retrieve("revenue profit margins", top_k=5, max_chars=10_000)
    assert hits == []  # those words only exist in document B
    hits = store.get("docB").retriever.retrieve("revenue profit margins", top_k=5, max_chars=10_000)
    assert hits and {h.chunk.document_id for h in hits} == {"docB"}


def test_retrieval_ranks_relevant_chunk_first_and_respects_limits():
    text = (
        "Cats sleep most of the day and enjoy warm places. " * 4
        + "The mitochondria produces energy for the cell using oxygen. " * 4
        + "Trains run on rails across the country every morning. " * 4
    )
    chunks = chunk_pages([PageText(1, text)], "d", 150, 20)
    retriever = BM25Retriever(chunks)
    hits = retriever.retrieve("What does the mitochondria produce?", top_k=2, max_chars=10_000)
    assert 1 <= len(hits) <= 2
    assert "mitochondria" in hits[0].chunk.text
    assert hits[0].score >= hits[-1].score
    limited = retriever.retrieve("mitochondria energy cell", top_k=10, max_chars=1)
    assert limited == []


def test_retrieval_with_only_stopwords_returns_nothing():
    chunks = chunk_pages([PageText(1, "Some ordinary content here.")], "d", 100, 10)
    assert BM25Retriever(chunks).retrieve("what is the", top_k=3, max_chars=1000) == []


def test_store_ttl_expiry_and_delete():
    now = [0.0]
    store = DocumentStore(ttl_seconds=10, max_documents=10, clock=lambda: now[0])
    store.add(_stored("old", "Some stored text for the old document."))
    now[0] = 11
    assert store.get("old") is None
    store.add(_stored("new", "Some stored text for the new document."))
    assert store.delete("new") is True
    assert store.delete("new") is False


def test_store_evicts_least_recently_used_when_full():
    store = DocumentStore(ttl_seconds=3600, max_documents=2)
    for name in ("a", "b"):
        store.add(_stored(name, f"Content of document {name} goes here."))
    store.get("a")  # "b" is now least recently used
    store.add(_stored("c", "Content of document c goes here."))
    assert store.get("b") is None
    assert store.get("a") is not None and store.get("c") is not None


> 🔧 **write_file** — """API tests using a fake model service (they do NOT test real model output)."""

from __future__ import annotations

import pytest

from tests.conftest import make_pdf

PDF_A = make_pdf(["The submarine explored the deep ocean trench near the volcano."])
PDF_B = make_pdf(["Quarterly revenue and profit margins improved across every region."])
UNKNOWN_ID = "0" * 32


def upload(client, data=PDF_A, name="a.pdf", content_type="application/pdf"):
    return client.post("/documents", files={"file": (name, data, content_type)})


def ask(client, doc_id, question):
    return client.post(f"/documents/{doc_id}/questions", json={"question": question})


def test_health_reports_actual_model_state(client, fake_model):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_ready"] is True
    fake_model.ready = False
    assert client.get("/health").json()["model_ready"] is False


def test_upload_valid_pdf(client):
    r = upload(client)
    assert r.status_code == 201
    body = r.json()
    assert len(body["document_id"]) == 32
    assert body["filename"] == "a.pdf"
    assert body["status"] == "processed"
    assert body["page_count"] == 1
    assert body["chunk_count"] >= 1
    assert "submarine" in body["text_preview"]
    assert body["word_count"] > 5


def test_upload_preview_is_bounded(client):
    long_pdf = make_pdf(["word " * 400])
    body = upload(client, long_pdf).json()
    assert len(body["text_preview"]) <= 500
    assert body["text_preview_truncated"] is True


def test_upload_wrong_type_rejected(client):
    r = upload(client, b"hello", "notes.txt", "text/plain")
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_file_type"


def test_upload_empty_rejected(client):
    r = upload(client, b"")
    assert r.status_code == 400 and r.json()["error"]["code"] == "empty_file"


def test_upload_oversized_rejected(client):
    r = upload(client, b"%PDF-" + b"0" * (1024 * 1024 + 10))
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_upload_malformed_rejected(client):
    r = upload(client, b"%PDF-1.4 not really a pdf")
    assert r.status_code == 422 and r.json()["error"]["code"] == "malformed_pdf"


def test_upload_without_text_rejected(client):
    r = upload(client, make_pdf(["", ""]))
    assert r.status_code == 422 and r.json()["error"]["code"] == "no_extractable_text"


def test_upload_missing_file_field(client):
    r = client.post("/documents")
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_question_answered_with_supporting_passages(client, fake_model):
    doc_id = upload(client).json()["document_id"]
    r = ask(client, doc_id, "What did the submarine explore?")
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "fake answer" and body["answer_generated"] is True
    assert body["model_name"] == "fake-model"
    assert body["inference_time_seconds"] >= 0
    passage = body["supporting_passages"][0]
    assert "submarine" in passage["text"]
    assert passage["page_number"] == 1
    assert passage["chunk_id"].startswith(doc_id)
    assert len(fake_model.calls) == 1


def test_question_retrieves_only_from_its_own_document(client, fake_model):
    id_a = upload(client, PDF_A, "a.pdf").json()["document_id"]
    id_b = upload(client, PDF_B, "b.pdf").json()["document_id"]

    r = ask(client, id_b, "revenue profit margins")
    assert r.status_code == 200
    assert all(p["chunk_id"].startswith(id_b) for p in r.json()["supporting_passages"])
    for _, passages in fake_model.calls:
        assert {p.chunk.document_id for p in passages} == {id_b}

    # Words that only exist in document B must not match anything in document A.
    r = ask(client, id_a, "revenue profit margins")
    body = r.json()
    assert body["answer_generated"] is False and body["answer"] is None
    assert body["supporting_passages"] == []
    assert len(fake_model.calls) == 1  # model not called again


def test_unknown_document_returns_404(client):
    r = ask(client, UNKNOWN_ID, "anything?")
    assert r.status_code == 404 and r.json()["error"]["code"] == "document_not_found"


def test_malformed_document_id_returns_400(client):
    r = ask(client, "not-a-valid-id", "anything?")
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_document_id"


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_rejected(client, question):
    doc_id = upload(client).json()["document_id"]
    r = ask(client, doc_id, question)
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


def test_missing_question_field_rejected(client):
    doc_id = upload(client).json()["document_id"]
    r = client.post(f"/documents/{doc_id}/questions", json={})
    assert r.status_code == 422


def test_model_not_ready_returns_503(client, fake_model):
    doc_id = upload(client).json()["document_id"]
    fake_model.ready = False
    r = ask(client, doc_id, "submarine?")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "model_not_ready"
    assert fake_model.calls == []


def test_delete_removes_document(client):
    doc_id = upload(client).json()["document_id"]
    r = client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert ask(client, doc_id, "submarine?").status_code == 404
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert client.get("/health").json()["documents_in_memory"] == 0


def test_delete_unknown_and_invalid_ids(client):
    assert client.delete(f"/documents/{UNKNOWN_ID}").status_code == 404
    assert client.delete("/documents/bad-id").status_code == 400


def test_cors_allows_configured_origin_only(client):
    ok = client.options(
        "/documents",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options(
        "/documents",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


def test_unexpected_exception_returns_generic_500(settings, fake_model):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings=settings, model_service=fake_model)
    client = TestClient(app, raise_server_exceptions=False)
    doc_id = upload(client).json()["document_id"]
    app.state.store.get(doc_id).retriever.retrieve = None  # break retrieval
    # Retrieval errors are wrapped in a controlled 500 without internal details.
    r = ask(client, doc_id, "submarine?")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "retrieval_failed"
    assert "Traceback" not in r.text


> 🔧 **write_file** — import pymupdf
import pytest

from app.document_processor import (
    EmptyFileError,
    FileTooLargeError,
    MalformedPDFError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    process_pdf,
    sanitize_filename,
)
from tests.conftest import make_pdf

MAX = 1024 * 1024


def test_valid_text_pdf_is_processed_page_by_page():
    pdf = make_pdf(
        ["The quarterly revenue grew strongly this year.", "Costs were reduced by automation."]
    )
    result = process_pdf("report.pdf", pdf, max_bytes=MAX)
    assert result.page_count == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "revenue" in result.pages[0].text
    assert "automation" in result.pages[1].text
    assert result.word_count > 5
    assert result.character_count == sum(len(p.text) for p in result.pages)
    assert result.empty_page_numbers == []


def test_blank_page_is_reported_but_document_still_accepted():
    pdf = make_pdf(["Some meaningful text about the project plan.", ""])
    result = process_pdf("a.pdf", pdf, max_bytes=MAX)
    assert result.page_count == 2
    assert result.empty_page_numbers == [2]


@pytest.mark.parametrize("name", ["notes.txt", "image.png", "pdf", None, ""])
def test_wrong_extension_is_rejected(name):
    with pytest.raises(UnsupportedFileTypeError):
        process_pdf(name, make_pdf(["hello world text here"]), max_bytes=MAX)


def test_pdf_extension_with_non_pdf_content_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        process_pdf("fake.pdf", b"this is not a pdf at all", max_bytes=MAX)


def test_empty_upload_is_rejected():
    with pytest.raises(EmptyFileError):
        process_pdf("a.pdf", b"", max_bytes=MAX)


def test_oversized_upload_is_rejected():
    with pytest.raises(FileTooLargeError):
        process_pdf("a.pdf", b"%PDF-" + b"0" * 200, max_bytes=100)


def test_malformed_pdf_is_rejected():
    with pytest.raises(MalformedPDFError):
        process_pdf("a.pdf", b"%PDF-1.4\n garbage that is not a real pdf", max_bytes=MAX)


def test_pdf_without_text_is_rejected():
    with pytest.raises(NoExtractableTextError):
        process_pdf("blank.pdf", make_pdf(["", ""]), max_bytes=MAX)


def test_image_only_pdf_is_rejected():
    doc = pymupdf.open()
    page = doc.new_page()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
    pix.set_rect(pix.irect, (200, 0, 0))
    page.insert_image(pymupdf.Rect(10, 10, 100, 100), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    with pytest.raises(NoExtractableTextError):
        process_pdf("scan.pdf", data, max_bytes=MAX)


def test_sanitize_filename_strips_directories():
    assert sanitize_filename("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("C:\\x\\y\\doc.pdf") == "doc.pdf"
    assert sanitize_filename(None) == "document.pdf"


> 🔧 **write_file** — """Tests that need the REAL configured model. Skipped unless MODEL_NAME_OR_PATH is set.

Run with:  MODEL_NAME_OR_PATH=<model> [ADAPTER_PATH=<dir>] pytest -m requires_model
"""

import os

import pytest

from app.chunking import chunk_pages
from app.config import Settings
from app.document_processor import PageText
from app.model_service import ModelService
from app.retrieval import BM25Retriever

pytestmark = [
    pytest.mark.requires_model,
    pytest.mark.skipif(
        not os.environ.get("MODEL_NAME_OR_PATH"), reason="MODEL_NAME_OR_PATH is not set"
    ),
]


def test_real_model_generates_a_nonempty_answer():
    service = ModelService(Settings(max_new_tokens=32))
    service.load()
    text = "The capital of the fictional country Zubrowka is Lutz. It lies in the north."
    chunks = chunk_pages([PageText(1, text)], "doc", 500, 50)
    passages = BM25Retriever(chunks).retrieve("What is the capital of Zubrowka?", 3, 5000)
    result = service.generate_answer("What is the capital of Zubrowka?", passages)
    assert result.answer
    assert result.generated_tokens > 0


### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && printf '%s\n' '# Secrets / local configuration' '.env' '.env.*' '!.env.example' '' '# Python' '__pycache__/' '*.py[cod]' '.venv/' 'venv/' '*.egg-info/' '.pytest_cache/' '.coverage' 'htmlcov/' '' '# Models and uploads must never be committed' 'models/' 'adapters/' 'uploads/' '*.safetensors' '*.bin' '*.pt' '*.gguf' '' '# OS / editors' '.DS_Store' '.idea/' '.vscode/' > .gitignore && python3 -m venv .venv && .venv/bin/pip install -q fastapi uvicorn python-multipart pydantic pydantic-settings pymupdf pytest httpx 2>&1 | tail -5; .venv/bin/pip list 2>/dev/null | grep -iE 'fastapi|pydantic|pymupdf|pytest|httpx|uvicorn|starlette'"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && rm -rf .venv; which uv virtualenv; ls -a; python3 -m pip install --user --break-system-packages -q virtualenv 2>&1 | tail -3; python3 -m virtualenv .venv 2>&1 | tail -3; ls .venv/bin | head"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && .venv/bin/pip install -q fastapi uvicorn python-multipart pydantic pydantic-settings pymupdf pytest httpx 2>&1 | tail -5; .venv/bin/pytest -q 2>&1 | tail -40"

### 🤖 Assistant

> 🔧 called: execute_command

All tests pass on the first run. The one warning says Starlette's TestClient now prefers `httpx2`, so I'll check that package and fix the warning.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && .venv/bin/pip install -q httpx2 2>&1 | tail -3; .venv/bin/pytest -q 2>&1 | tail -8; .venv/bin/pip list 2>/dev/null | grep -iE '^(fastapi|starlette|pydantic|pydantic-settings|pymupdf|httpx2|httpx|uvicorn|pytest|python-multipart) '"

### 🤖 Assistant

> 🔧 called: execute_command

`httpx2` removes the warning, so I'll swap it into `requirements.txt`. Then I'll try to verify `model_service.py` with a tiny public test model, since the fake model service doesn't exercise it.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && sed -i 's/^httpx>=0.27/httpx2>=2.0/' requirements.txt && cat requirements.txt && (.venv/bin/pip install -q torch transformers peft accelerate --extra-index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3); .venv/bin/pip list 2>/dev/null | grep -iE '^(torch|transformers|peft|accelerate) '"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && cat > /tmp/smoke.py <<'EOF'
import tempfile, os
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from app.config import Settings
from app.model_service import ModelService
from app.chunking import chunk_pages
from app.document_processor import PageText
from app.retrieval import BM25Retriever

MID = 'hf-internal-testing/tiny-random-LlamaForCausalLM'
base = AutoModelForCausalLM.from_pretrained(MID)
peft_model = get_peft_model(base, LoraConfig(r=2, target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM'))
d = tempfile.mkdtemp(); peft_model.save_pretrained(d)

chunks = chunk_pages([PageText(1, 'The capital of Zubrowka is Lutz. It lies in the north.')], 'doc', 500, 50)
passages = BM25Retriever(chunks).retrieve('capital of Zubrowka', 3, 5000)
for adapter in (None, d):
    s = Settings(_env_file=None, model_name_or_path=MID, adapter_path=adapter, max_new_tokens=8, model_device='cpu')
    svc = ModelService(s)
    assert not svc.is_ready
    svc.load()
    r = svc.generate_answer('What is the capital?', passages)
    print('adapter' if adapter else 'base', svc.status['adapter_loaded'], r.generated_tokens, r.prompt_tokens, r.passages_used, repr(r.answer[:30]))
EOF
.venv/bin/python /tmp/smoke.py 2>&1 | tail -15"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && PYTHONPATH=. .venv/bin/python /tmp/smoke.py 2>&1 | tail -15"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && (.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765 >/tmp/uv.log 2>&1 &) ; sleep 6; curl -s http://127.0.0.1:8765/health; echo; python3 - <<'EOF'
import fitz_unused if False else None
EOF
.venv/bin/python -c \"
import pymupdf
d=pymupdf.open();p=d.new_page();p.insert_text((72,72),'The submarine explored the deep ocean trench near the volcano.');d.save('/tmp/t.pdf')\"
curl -s -F file=@/tmp/t.pdf http://127.0.0.1:8765/documents | tee /tmp/up.json; echo
ID=\$(.venv/bin/python -c \"import json;print(json.load(open('/tmp/up.json'))['document_id'])\")
curl -s -X POST -H 'Content-Type: application/json' -d '{\"question\":\"What did the submarine explore?\"}' http://127.0.0.1:8765/documents/\$ID/questions; echo
curl -s -X DELETE http://127.0.0.1:8765/documents/\$ID; echo; pkill -f 'uvicorn app.main' ; grep -iE 'error|warn' /tmp/uv.log | head"

### 🤖 Assistant

> 🔧 called: write_file, write_file

PowerShell quoting broke that command. I'll put the live-server check in a script file instead.

> 🔧 **write_file** — import pymupdf
d = pymupdf.open()
p = d.new_page()
p.insert_text((72, 72), "The submarine explored the deep ocean trench near the volcano.")
d.save("/tmp/t.pdf")


> 🔧 **write_file** — #!/usr/bin/env bash
# Manual end-to-end check against a running server (default http://127.0.0.1:8000).
# Usage: scripts/smoke_test.sh path/to/file.pdf "Your question?"
set -euo pipefail
BASE="${BASE_URL:-http://127.0.0.1:8000}"
PDF="${1:?usage: smoke_test.sh file.pdf \"question\"}"
QUESTION="${2:-What is this document about?}"

echo "--- GET /health"
curl -sS "$BASE/health"; echo

echo "--- POST /documents"
UPLOAD=$(curl -sS -F "file=@${PDF};type=application/pdf" "$BASE/documents")
echo "$UPLOAD"
DOC_ID=$(printf '%s' "$UPLOAD" | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')

echo "--- POST /documents/$DOC_ID/questions"
curl -sS -X POST -H 'Content-Type: application/json' \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$QUESTION")" \
  "$BASE/documents/$DOC_ID/questions"; echo

echo "--- DELETE /documents/$DOC_ID"
curl -sS -X DELETE "$BASE/documents/$DOC_ID"; echo


### 🤖 Assistant

> 🔧 called: execute_command

I'll start the server with the tiny test model (via env vars only, no `.env` file) and run the smoke script against it.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && chmod +x scripts/smoke_test.sh && .venv/bin/python .mkpdf_tmp.py && rm .mkpdf_tmp.py && (MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM MODEL_DEVICE=cpu MAX_NEW_TOKENS=8 nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8765 >/tmp/uv.log 2>&1 &) ; sleep 12; BASE_URL=http://127.0.0.1:8765 scripts/smoke_test.sh /tmp/t.pdf 'What did the submarine explore?'; pkill -f 'uvicorn app.main'; echo; echo LOG; grep -ciE 'traceback' /tmp/uv.log"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && (nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8766 >/tmp/uv2.log 2>&1 &) ; sleep 6; curl -s http://127.0.0.1:8766/health; echo; curl -s -F file=@/tmp/t.pdf http://127.0.0.1:8766/documents | cut -c1-80; echo; ID=\$(curl -s -F file=@/tmp/t.pdf http://127.0.0.1:8766/documents | .venv/bin/python -c 'import json,sys;print(json.load(sys.stdin)[\"document_id\"])'); curl -s -X POST -H 'Content-Type: application/json' -d '{\"question\":\"submarine?\"}' http://127.0.0.1:8766/documents/\$ID/questions; echo; pkill -f 'uvicorn app.main'; true"

### 🤖 Assistant

> 🔧 called: execute_command

The live run with the tiny model worked end to end. I still need to check the unconfigured case, where the server should start and report the problem clearly. I'll reuse the smoke script for that.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && (nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8766 >/tmp/uv2.log 2>&1 &) ; sleep 6; BASE_URL=http://127.0.0.1:8766 scripts/smoke_test.sh /tmp/t.pdf 'submarine?' 2>&1 | cut -c1-400; pkill -f 'uvicorn app.main'; grep -i 'not available' /tmp/uv2.log"

### 🤖 Assistant

> 🔧 called: write_file

Unconfigured behaviour is correct: the server starts, `/health` reports `model_ready: false` with the reason, and questions return a clean 503. Now the README.

> 🔧 **write_file** — # SLM Document Analysis Backend

FastAPI backend for the M.Tech project *"A Study of Parameter-Efficient Fine-Tuning of Small
Language Models (SLMs) for Document Analysis"*.

It performs **inference only**: upload a text-based PDF, split it into chunks, ask questions, and
get answers from your configured Small Language Model (base model + optional LoRA adapter). It does
**not** train or fine-tune anything and is unrelated to the training-data notebook.

## Project structure

```
slm-document-analysis-backend/
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app, endpoints, error handlers, CORS
│   ├── config.py              # Settings read from environment variables / .env
│   ├── errors.py              # ApiError (uniform error responses)
│   ├── schemas.py             # Pydantic request/response models
│   ├── document_processor.py  # PDF validation + page-by-page text extraction (PyMuPDF)
│   ├── chunking.py            # Page-aware overlapping chunking
│   ├── retrieval.py           # BM25 lexical retrieval (per document)
│   ├── prompting.py           # Prompt construction
│   ├── model_service.py       # Model + LoRA loading and generation
│   └── storage.py             # Temporary in-memory document store (TTL + max count)
├── tests/
│   ├── conftest.py            # PDF generator + fake model service
│   ├── test_document_processor.py
│   ├── test_chunking.py       # chunking, retrieval isolation, storage
│   ├── test_api.py
│   └── test_model_integration.py   # needs a real model; skipped by default
├── scripts/smoke_test.sh      # manual end-to-end check with curl
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## 1. Set up (macOS)

Requires Python 3.11–3.13 (any version supported by your PyTorch/Transformers install).

```bash
cd slm-document-analysis-backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 2. Configure your model (REQUIRED)

No model location is assumed. Edit `.env`:

| Variable | What to put there |
|---|---|
| `MODEL_NAME_OR_PATH` | The **same base model** you fine-tuned: a Hugging Face id or a local folder. |
| `ADAPTER_PATH` | Folder with your saved LoRA adapter (contains `adapter_config.json`). Leave empty to run the base model alone. |
| `TOKENIZER_NAME_OR_PATH` | Optional; only if the tokenizer is stored elsewhere. |
| `MODEL_DEVICE` | `auto` (cuda → mps → cpu), `cpu`, `mps` (Apple Silicon GPU) or `cuda`. |
| `MODEL_DTYPE` | `auto` (float32 on CPU, float16 otherwise), or `float32` / `float16` / `bfloat16`. |
| `MERGE_ADAPTER` | `true` merges LoRA weights into the base model (slightly faster inference). |

If `MODEL_NAME_OR_PATH` is empty the server still starts, `/health` returns `model_ready: false` with
the reason, and question requests return HTTP 503 `model_not_ready`. It never substitutes another model.

**Prompt format.** The prompt text is in `app/prompting.py`. The model's own chat template is used
automatically if its tokenizer has one. If your adapter was trained with a particular prompt layout,
change `prompting.py` to match it; mismatched prompts degrade answer quality.

All other settings (chunk size/overlap, top-k, context length, max answer tokens, upload limit,
CORS origins, document TTL) are documented in `.env.example`.

## 3. Run

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>. Do **not** use `--workers N` > 1: documents live in
one process's memory, so other workers would not see them.

Quick manual check (server running):

```bash
scripts/smoke_test.sh path/to/file.pdf "What is this document about?"
```

## 4. Run the tests

```bash
pytest                          # unit + API tests; no model download needed
MODEL_NAME_OR_PATH=<model> [ADAPTER_PATH=<dir>] pytest -m requires_model   # real-model test
```

The API tests use a **fake model service**; they verify the HTTP behaviour, not real model output.

## 5. API reference

All errors use one shape: `{"error": {"code": "<machine_code>", "message": "<text>"}}`.

### `GET /health`
```json
{"status":"ok","model_ready":true,"model_name":"<base> + LoRA adapter '<dir>'",
 "adapter_loaded":true,"device":"mps","model_error":null,"documents_in_memory":1}
```
`model_ready` reflects whether the model and tokenizer are actually loaded.

### `POST /documents` (multipart/form-data, field `file`) → 201
```json
{"document_id":"5911117fac36410f8bc8751b4d213979","filename":"t.pdf","status":"processed",
 "page_count":1,"character_count":63,"word_count":10,"chunk_count":1,"empty_page_numbers":[],
 "text_preview":"The submarine explored ...","text_preview_truncated":false}
```
(`text_preview` is capped at `TEXT_PREVIEW_CHARS`.) Errors:

| Status | code | Cause |
|---|---|---|
| 400 | `empty_file` | zero-byte upload |
| 413 | `file_too_large` | larger than `MAX_UPLOAD_MB` |
| 415 | `unsupported_file_type` | not a `.pdf` or no PDF signature |
| 422 | `malformed_pdf` | corrupt / password-protected PDF |
| 422 | `no_extractable_text` | no meaningful text (e.g. scanned PDF; OCR not supported) |
| 422 | `validation_error` | missing `file` field |

### `POST /documents/{document_id}/questions`
Request: `{"question": "What is the main conclusion?"}` (1–2000 chars, not blank).

Response 200 (answer generated):
```json
{"document_id":"…","question":"…","answer":"<model output>","answer_generated":true,"message":null,
 "model_name":"…","inference_time_seconds":1.25,"prompt_tokens":512,"generated_tokens":40,
 "supporting_passages":[{"text":"<retrieved text>","page_number":3,"chunk_id":"<docid>-c0007","retrieval_score":2.31}],
 "disclaimer":"Answers are generated by a small language model and may be incorrect. …"}
```
`retrieval_score` is a BM25 ranking value, **not** a probability or confidence.

If no chunk matches the question the model is **not called**: `answer` is `null`,
`answer_generated` is `false`, `supporting_passages` is `[]` and `message` explains why.

| Status | code | Cause |
|---|---|---|
| 400 | `invalid_document_id` | malformed ID |
| 404 | `document_not_found` | unknown, deleted, expired, or server restarted |
| 422 | `validation_error` | empty/missing question |
| 503 | `model_not_ready` / `model_not_configured` | model not loaded |
| 504 | `inference_timeout` | exceeded `GENERATION_TIMEOUT_SECONDS` |
| 500 | `inference_failed` / `retrieval_failed` / `internal_error` | other failures (details only in server logs) |

### `DELETE /documents/{document_id}`
```json
{"document_id":"…","deleted":true,"message":"The document and its chunks were removed from in-memory storage."}
```
400 `invalid_document_id`, 404 `document_not_found`.

## 6. Frontend example (React / Vite, origin `http://localhost:5173`)

```js
const API = "http://127.0.0.1:8000";

async function api(path, options) {
  const res = await fetch(`${API}${path}`, options);
  const body = await res.json();
  if (!res.ok) throw new Error(body.error?.message ?? "Request failed");
  return body;
}

export const health = () => api("/health");

export function uploadPdf(file) {
  const form = new FormData();
  form.append("file", file);                       // field name must be "file"
  return api("/documents", { method: "POST", body: form }); // do not set Content-Type manually
}

export const ask = (documentId, question) =>
  api(`/documents/${documentId}/questions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });

export const deleteDocument = (documentId) =>
  api(`/documents/${documentId}`, { method: "DELETE" });
```

Workflow: `health()` → `uploadPdf(file)` (keep `document_id`) → `ask(id, q)` as many times as needed
(show `answer` and `supporting_passages`) → `deleteDocument(id)`. Other origins must be added to
`CORS_ORIGINS`.

## 7. Design notes

* **Extraction:** PyMuPDF, page by page; page numbers are 1-based. Text-based PDFs only. **No OCR.**
* **Chunking:** character windows (`CHUNK_SIZE`, `CHUNK_OVERLAP`) per page, cut at whitespace so words
  are not split; each chunk is an exact slice of the page text. The chunk end always advances, so it
  cannot loop forever or produce duplicate-only chunks. Chunk IDs: `<document_id>-c0000`, …
* **Retrieval:** BM25 (lexical), implemented in ~60 lines with no extra model. Chosen because it adds
  no memory/download cost next to the SLM and is easy to explain. Limitation: it matches words, not
  meaning, so paraphrased questions can miss relevant passages. **No semantic/embedding retrieval is
  implemented.** The `Retriever` protocol in `retrieval.py` is where an upgrade would go.
  Each document has its own retriever, so a question can never retrieve another document's chunks.
* **Context limits:** at most `RETRIEVAL_TOP_K` chunks and `MAX_CONTEXT_CHARS` characters are
  retrieved; the model service additionally drops lowest-ranked passages (then truncates) to fit
  `MAX_INPUT_TOKENS` and the model's context window. Only the passages actually used are returned as
  `supporting_passages`.
* **Generation:** only newly generated tokens are decoded. One generation runs at a time.
  `GENERATION_TIMEOUT_SECONDS` is enforced through `max_time`, which stops generation cooperatively.
* **Storage:** in-memory only. It is **temporary**, **not shared between processes**, and **lost on
  restart**. Documents expire after `DOCUMENT_TTL_MINUTES` of inactivity and the least recently used
  are evicted beyond `MAX_DOCUMENTS` (cleanup runs lazily on requests). Uploaded files are never
  written to disk; the filename is only a display label.
* **Not guaranteed correct:** answers come from a small model and may be wrong; every response carries
  a disclaimer and the supporting passages so users can verify.

## 8. Can the model run locally?

It depends on your model, which I could not inspect (see below).

* **Apple Silicon MacBook Air:** models up to roughly 1–3B parameters are usually workable with
  `MODEL_DEVICE=mps` (or `cpu`) and `float16`/`float32`, but memory is shared (8–16 GB) and generation
  is slow. 7B models will generally not fit in float16; use a smaller model.
* **CUDA-only setups** (e.g. a model/adapter trained with 4-bit `bitsandbytes`) do not run on macOS.
  Merge/load the adapter on a non-quantised base model, or run inference elsewhere.
* **Remote inference (NOT implemented).** If the model is too large, inference must run on a GPU host
  (e.g. Google Colab). The local FastAPI app would then need a new class with the same interface as
  `ModelService` (`is_ready`, `status`, `generate_answer`) that calls a remote HTTP service, passed to
  `create_app(model_service=...)`. Colab's `localhost` is **not** reachable from your Mac; you would
  need a tunnel (e.g. ngrok/cloudflared) and HTTPS plus a secret token kept in the backend's
  environment variables, never in frontend code. None of this exists in this project yet.

## 9. Known limitations

* No OCR (scanned PDFs are rejected), no tables/figures understanding.
* Lexical retrieval only; no re-ranking or semantic search.
* Single process, in-memory, no authentication, no persistence, no streaming responses.
* Requests are processed one generation at a time; long generations queue.


### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — slm-document-analysis-backend/README.md

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && .venv/bin/pytest -q -rs 2>&1 | tail -6; pkill -f 'uvicorn app.main'; find . -path ./.venv -prune -o -type f -not -path '*/__pycache__/*' -not -path './.pytest_cache/*' -print | sort"

### 🤖 Assistant

The backend is built in `slm-document-analysis-backend/`, and its 50 tests pass (run on Linux under WSL, not macOS). One more test is skipped: it needs a real model, so I haven't run it. I couldn't read your training project in `../Mtech`, so I don't know your base model or adapter. `MODEL_NAME_OR_PATH` and `ADAPTER_PATH` are blank placeholders you have to fill in.

## Files created
- **`app/`**:
  - `main.py` (API, error handlers, CORS)
  - `config.py`
  - `schemas.py`
  - `errors.py`
  - `document_processor.py` (PyMuPDF)
  - `chunking.py`
  - `retrieval.py` (BM25)
  - `prompting.py`
  - `model_service.py` (Transformers + PEFT)
  - `storage.py`
- **`tests/`**:
  - `test_document_processor.py`
  - `test_chunking.py` (also covers retrieval and storage)
  - `test_api.py`
  - `test_model_integration.py`
  - `conftest.py`
- **Project files:**
  - `requirements.txt`
  - `.env.example`
  - `.gitignore`
  - `pytest.ini`
  - `README.md` (setup, endpoint formats, React example)
  - `scripts/smoke_test.sh` (curl check against a running server)

## How the API works
- **`GET /health`** reports `model_ready` from the real model state. With no model configured, the server still starts and reports why.
- **`POST /documents`** validates the PDF, extracts text page by page and chunks it. It stores the chunks in memory and returns the document ID, counts and a preview capped at 500 characters.
- **`POST /documents/{id}/questions`** retrieves chunks only from that document and builds the prompt. It then calls the model and returns the answer with supporting passages (text, page, chunk ID). If no chunk matches, the model isn't called and the response says so.
- **`DELETE /documents/{id}`** removes the document from memory.
- **Errors:** all errors use `{"error": {"code", "message"}}`, with a specific status for each failure case you listed.

## What I verified
- **Unit and API tests:** they use generated PDFs and a fake model service.
- **Isolation and storage:** tests confirm a question can't retrieve chunks from another document, and that delete and unknown IDs behave correctly.
- **Real model path:** I ran the real `ModelService` on a tiny random public Llama test model, both with and without a freshly made LoRA adapter. I also ran the live server with `scripts/smoke_test.sh`.
- **No-model start:** the server starts without a model and returns a clean 503 on questions.
- **Not verified:** the tiny model produces gibberish, so none of this says anything about answer quality. Nothing was tested on a Mac or with your own model.

## What you need to change
1. Copy `.env.example` to `.env`.
2. Set `MODEL_NAME_OR_PATH` to the base model you fine-tuned.
3. Set `ADAPTER_PATH` to your LoRA adapter folder, or leave it empty.
4. Optionally set `MODEL_DEVICE` (`mps`, `cpu` or `auto`).
5. Check `app/prompting.py` against the prompt format you used in training.

## Run on macOS
```bash
cd slm-document-analysis-backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit it
uvicorn app.main:app --host 127.0.0.1 --port 8000
pytest                      # tests
scripts/smoke_test.sh file.pdf "question?"
```

## Local or remote inference
- **Small models:** about 1–3B parameters are usually workable on a MacBook Air with `mps` or `cpu`, but slow. 7B models generally won't fit.
- **CUDA-only models:** models trained with 4-bit `bitsandbytes` won't run on macOS.
- **Remote inference is not implemented.** `main.py` only needs an object with the same interface as `ModelService`, but the Colab and tunnel pieces would still have to be built. The README explains the networking caveats.

## Incomplete or limited
- No OCR: scanned PDFs are rejected.
- BM25 matches words, not meaning, so paraphrased questions can miss passages. There is no semantic retrieval.
- Run it as a single process only: documents are in memory and lost on restart.
- If your adapter used a specific prompt template, you must match it in `prompting.py`.

## Housekeeping
The `.venv` I created for testing is inside the project folder and is git-ignored. It uses CPU PyTorch from WSL, so delete it and create a fresh one on your Mac.

---

### 👤 User

# Project: SLM-Based Document Analysis Web Application

## 1. Project Overview

I am developing an M.Tech project titled:

**"A Study of Parameter-Efficient Fine-Tuning of Small Language Models (SLMs) for Document Analysis."**

I need a working proof-of-concept web application that allows users to upload a PDF document, extract and process its text, ask questions about the document, and receive answers generated by a Small Language Model (SLM).

I want you to generate the complete application, including:

1. A frontend web interface.
2. A Python backend that runs in Google Colab.
3. API endpoints that connect the frontend to the backend.
4. Document text extraction and chunking.
5. Question answering using the SLM.
6. Instructions for running and testing the complete application.

I am primarily a Python backend developer and have limited frontend development experience. Therefore, generate clean, simple, well-documented code and provide exact, beginner-friendly setup instructions.

Do not only provide sample code or a design mockup. Generate the actual files and working implementation wherever the environment supports file creation.

## 2. Overall Architecture

Use the following architecture:

Frontend:
- React
- TypeScript
- HTML
- CSS
- Vite as the development server

Backend:
- Python
- FastAPI
- Google Colab for running the backend and model inference

Model:
- A Small Language Model suitable for document question answering.
- Support my existing fine-tuned model and saved LoRA adapter, if available.
- Allow the model configuration to be changed without rewriting the application.

The frontend will run on my MacBook Air.

The Python backend, document processing and model inference will run in Google Colab.

The frontend will communicate with the backend through HTTP API requests.

Do not assume that a frontend running on my MacBook can directly access Python variables or functions inside Google Colab. Implement an appropriate API-based connection.

If the chosen approach requires a public tunnel or externally accessible endpoint, explain how to configure it safely. Do not expose API credentials, model credentials or other secrets in the frontend.

## 3. Frontend Design

Create a professional, modern and easy-to-use interface for an academic project demonstration.

Application title:

SLM Document Analysis

Subtitle:

Upload a document, ask questions, and explore how Small Language Models can understand document content.

Add a small badge displaying:

Proof of Concept

Use a clean, light-themed design with:
- A white or light-grey background.
- Navy or dark-grey headings.
- Blue or indigo buttons.
- Rounded cards.
- Clear spacing.
- Readable fonts.
- Simple icons where appropriate.
- Responsive layout suitable for a laptop screen.

Avoid excessive animations, unnecessary decorative elements and complicated navigation.

The application should have the following sections.

### Section A: Header

Display:
- Application title: SLM Document Analysis.
- Subtitle explaining the application.
- Proof of Concept badge.

### Section B: Document Upload

Display a card titled:

Step 1: Upload Your Document

Include:
- A drag-and-drop upload area.
- A Browse Files button.
- Support for PDF files.
- Selected filename.
- File size.
- Upload and Process Document button.

Validate the selected file and display useful error messages for invalid file types or empty files.

After the user clicks Upload and Process Document:
1. Send the PDF to the backend API.
2. Show a loading indicator.
3. Display the current processing status.
4. Wait for successful completion.
5. Display a success message when text extraction and processing finish.

Show a warning that the initial version supports text-based PDFs. Scanned PDFs may require OCR, which can be implemented later if it is not already available in the existing backend.

### Section C: Extracted Text Preview

Display a card titled:

Step 2: Extracted Document Text

After processing succeeds, display:
- A preview of the actual extracted text.
- Total character count.
- Total word count, if practical.
- An expand/collapse option.
- A Copy Text button.

Use actual text returned by the backend.

Do not display invented or hardcoded document text.

If text extraction fails or returns empty text, display an appropriate error message and prevent question answering until the problem is resolved.

The backend should split the extracted text into manageable chunks for model processing.

The chunking implementation should:
- Preserve the original text.
- Use configurable chunk size and overlap.
- Avoid unnecessarily splitting text into individual words or sentences without context.
- Handle long documents.
- Avoid sending an entire long document to the model when it exceeds the model's context window.

Display the number of chunks created.

### Section D: Ask a Question

Display a card titled:

Step 3: Ask a Question

Include:
- A multiline question input box.
- A placeholder such as "Ask a question about your uploaded document..."
- A Generate Answer button.

Provide a few optional example questions, but do not hardcode their answers.

The user should be able to enter any relevant question.

Enable question answering only after a document has been processed successfully.

When the user clicks Generate Answer:
1. Send the question and the identifier of the processed document to the backend.
2. The backend should retrieve the relevant chunks from that document.
3. Prepare a prompt containing the relevant document context and the user's question.
4. Pass the prompt to the configured SLM.
5. Generate the answer.
6. Return the answer and any available supporting evidence to the frontend.

Do not require the user to upload the same PDF again for every question.

Allow the user to ask multiple questions about the same uploaded document.

### Section E: Generated Answer

Display a card titled:

Step 4: Generated Answer

Display:
- The actual answer returned by the backend.
- The question asked.
- The model used, if the backend returns this information.
- Inference time, if measured by the backend.
- A Copy Answer button.

If the backend returns the relevant document passages used to generate the answer, display them in a separate section titled:

Supporting Evidence from the Document

Show only evidence actually returned by the backend.

Do not fabricate supporting passages, citations, confidence scores, model names or performance metrics.

The model should be instructed to answer using the document context and to state when the answer cannot be found in the available text.

### Section F: Reset

Provide a Start New Document or Reset button.

It should clear the current document, question and answer from the frontend.

If a backend API is available to clear server-side session data, call it appropriately. Do not claim server-side data has been deleted if it has only been cleared from the browser.

## 4. Backend Notebook for Google Colab

Create a complete Google Colab notebook named:

slm_document_analysis_backend.ipynb

The notebook should contain clearly organised cells with Markdown explanations and Python code.

Use the following structure.

### Cell 1: Install Dependencies

Install only the packages required by the implementation.

Potential packages include:
- FastAPI
- Uvicorn
- python-multipart
- PyMuPDF or another suitable PDF text-extraction library
- Transformers
- PyTorch
- PEFT, if required for LoRA
- Other dependencies genuinely needed by the selected model

Check dependency compatibility and avoid unnecessary installations.

### Cell 2: Configuration

Create configuration variables for:
- Base model identifier or local model path.
- LoRA adapter path, if applicable.
- Device and precision configuration.
- Chunk size.
- Chunk overlap.
- Maximum number of context chunks.
- Maximum answer length.
- API configuration.

Do not invent a model path or claim that an adapter exists.

If the existing project already defines a base model and adapter, reuse those definitions.

If any configuration value is unknown, clearly mark it for me to fill in.

### Cell 3: Load the Model

Implement model loading.

Prefer loading my existing fine-tuned SLM and LoRA adapter if they are available.

If an adapter is used:
- Load the correct base model.
- Load the corresponding adapter using the appropriate PEFT mechanism.
- Put the model into inference mode.
- Use an appropriate device and precision configuration.
- Avoid loading the model repeatedly for each question.

If the adapter is stored in Google Drive, explain how to mount Drive and configure the path.

If I have not provided a model identifier or adapter path, do not silently invent one. Use a clearly documented placeholder or ask me to provide the missing configuration.

Keep model loading separate from document upload and question-answering functions.

### Cell 4: PDF Text Extraction

Implement a Python function that:
- Accepts an uploaded PDF.
- Extracts text page by page.
- Handles extraction errors.
- Detects when the PDF contains no extractable text.
- Returns the extracted text and useful document metadata.

Use a text-extraction library appropriate for digital PDFs.

Do not silently claim OCR support if OCR is not implemented.

### Cell 5: Text Chunking

Implement a reusable function that divides extracted text into manageable chunks.

Requirements:
- Configurable chunk size and overlap.
- Preserve document context.
- Avoid empty chunks.
- Assign a unique identifier to each processed document.
- Associate every chunk with its document identifier and page metadata where available.

Do not introduce a vector database unless it is necessary.

For the initial version, keep the processed document and chunks in memory on the backend.

Clearly explain that in-memory storage is temporary and can be lost when the Colab session restarts.

### Cell 6: Relevant Chunk Selection

Implement a simple, practical method for selecting relevant document chunks for each question.

Prefer a lightweight approach that can run in Google Colab.

For example, use a suitable embedding model and cosine similarity if the dependencies and runtime permit it. Alternatively, use another justified retrieval method if that is more practical.

Make the retrieval implementation modular so that it can be replaced later.

Do not claim that semantic retrieval is implemented unless the code actually performs it.

Ensure that retrieved chunks belong to the document associated with the user's question.

Limit the total context sent to the SLM to fit within its context window.

If no suitable chunks can be found, handle the situation gracefully rather than silently passing unrelated content.

### Cell 7: Prompt Construction

Create a reusable function that constructs the model prompt using:
- The selected document context.
- The user's question.
- Clear instructions to answer based on the supplied context.

Use an instruction similar to:

"You are a document question-answering assistant. Answer the user's question using the supplied document context. Do not invent facts that are not supported by the context. If the answer cannot be found in the available context, clearly state that you could not find the answer in the document."

Adapt the prompt format to the selected model's actual chat template or instruction format.

Do not assume every model uses the same prompt template.

### Cell 8: Model Inference

Implement a reusable function that:
- Accepts the constructed prompt.
- Calls the loaded model.
- Generates an answer.
- Measures inference time.
- Handles generation errors.
- Returns the generated answer and useful metadata.

Use the actual model configured in the notebook.

Do not return a hardcoded answer or substitute an unrelated external model.

Ensure that the answer contains newly generated tokens rather than accidentally displaying the entire input prompt.

### Cell 9: FastAPI Application

Create a FastAPI application with the following endpoints.

#### Endpoint 1: Health Check

GET /health

Purpose:
- Confirm that the API server is running.
- Return basic status information.
- Report model readiness if this can be determined reliably.

#### Endpoint 2: Upload and Process Document

POST /documents

Purpose:
- Accept a PDF file.
- Validate the file.
- Extract its text.
- Split the text into chunks.
- Store the processed document and chunks in backend memory.
- Return a unique document ID and processing metadata.

Return information such as:
- document_id
- filename
- extraction status
- number of pages, if available
- word count
- character count
- chunk count
- a preview of the extracted text

Return appropriate HTTP error responses for invalid files, extraction failures and empty extracted text.

Do not send the entire document back to the frontend if it is too large. Return a bounded preview and expose the complete text only if needed by the UI.

#### Endpoint 3: Ask a Question

POST /documents/{document_id}/questions

Accept a JSON request containing the user's question.

Purpose:
1. Validate the document ID.
2. Validate the question.
3. Find the corresponding processed document.
4. Retrieve relevant chunks from that document.
5. Construct the model prompt.
6. Run inference using the configured SLM.
7. Return the generated answer and metadata.

The response should contain fields similar to:
- document_id
- question
- answer
- model_name, if known
- inference_time_seconds
- supporting_passages, if available

Each supporting passage should contain actual retrieved text and page information when available.

Do not return fabricated evidence or fabricated model metadata.

#### Endpoint 4: Reset Document

DELETE /documents/{document_id}

Purpose:
- Remove the specified document and its chunks from in-memory storage.
- Return a meaningful response.

If the implementation does not support this operation, do not expose a fake endpoint.

### Cell 10: CORS Configuration

Configure FastAPI CORS so that the local frontend can call the backend.

Use a configurable list of allowed frontend origins.

For local development, allow the actual local frontend origin, such as the Vite development server origin.

If a public tunnel is used, explain how the allowed origin should be configured.

Do not use unrestricted CORS as a substitute for proper configuration.

### Cell 11: Start the API Server

Provide a practical way to start the FastAPI server inside Google Colab.

The server must run while the notebook session remains active.

If external access is needed, document a suitable tunnelling approach and how to configure it.

Do not assume that localhost inside Colab is directly accessible from my MacBook.

If a tunnel requires an authentication token, obtain it through a secure configuration mechanism and never hardcode it in source code or expose it to the frontend.

Explain how to stop and restart the server.

If the notebook requires special handling to run a persistent server, include the correct implementation.

## 5. Frontend and Backend Integration

Create a central frontend configuration value for the backend API base URL.

For example, use a Vite environment variable such as:

VITE_API_BASE_URL

Do not hardcode the Colab tunnel URL throughout the application.

The frontend must call the actual backend endpoints.

The intended flow is:

1. Check backend availability using GET /health.
2. Upload the PDF using POST /documents.
3. Receive the document ID and extraction metadata.
4. Display the extraction status and text preview.
5. Allow the user to enter a question.
6. Send the question to POST /documents/{document_id}/questions.
7. Display the actual answer and supporting passages returned by the backend.
8. Allow additional questions about the same document.
9. Reset the current document using the delete endpoint if available.

Handle network failures, unavailable Colab sessions, invalid responses and inference errors.

Display helpful messages instead of leaving the user with a blank screen.

Do not create a frontend that only looks functional but uses fake responses.

## 6. Security and Reliability

Implement reasonable protections for a proof of concept:
- Validate uploaded file types and sizes.
- Handle malformed PDFs.
- Avoid logging full document contents unnecessarily.
- Do not expose model credentials or tunnel tokens.
- Avoid accepting arbitrary filesystem paths from frontend requests.
- Validate document IDs and question input.
- Handle missing models and adapters explicitly.
- Do not claim successful processing when an operation fails.

Document the limitations of using in-memory storage and a temporary Colab session.

Do not add a database, authentication system, distributed workers, Docker, Kubernetes or other infrastructure unless genuinely necessary.

## 7. Important Model and Research Requirements

This application is a proof of concept for my academic research.

The implementation must use the configured SLM and must not silently replace it with a different model.

Do not claim that the model was fine-tuned on my uploaded PDF. Uploading a document should only process it for inference.

The saved LoRA adapter, if configured, should be loaded for inference. Model fine-tuning is a separate offline experiment.

Do not claim that the application proves the model is better than another model unless an actual comparison experiment has been performed.

Do not display invented:
- Accuracy scores.
- Evaluation metrics.
- Confidence scores.
- Model names.
- Supporting evidence.
- Answers.
- Processing times.

For the initial proof of concept, prioritise a working question-answering pipeline over a complicated evaluation dashboard.

## 8. Optional Features

Only implement these if the core workflow is working:
- Download or copy the generated answer.
- Display the number of chunks.
- Display supporting passages.
- Compare a zero-shot base model with the LoRA fine-tuned model.
- Show a simple processing timeline.

Model comparison must invoke the actual configured models and display their real outputs. Do not implement a fake comparison screen.

OCR for scanned PDFs and advanced retrieval can be added later if time permits.

## 9. Deliverables

Generate the following deliverables:

1. A complete React frontend project.
2. A complete Google Colab notebook named slm_document_analysis_backend.ipynb.
3. A requirements.txt file for the Python backend.
4. A frontend environment-variable example file, such as .env.example.
5. A README.md file containing setup instructions.
6. A list of the actual API endpoints and request/response formats.
7. Exact terminal commands for starting the frontend on macOS.
8. Exact Colab instructions for installing dependencies, configuring the model, starting the API server and exposing it to the frontend.
9. A complete end-to-end test procedure.
10. A list of any configuration values or model paths that I must supply.

If you can create downloadable files directly, create the actual files instead of merely printing snippets.

## 10. Required End-to-End Testing

Verify the intended workflow:

1. Start the backend in Google Colab.
2. Confirm that the backend health endpoint works.
3. Configure the frontend with the correct backend API URL.
4. Start the frontend on my MacBook.
5. Open the UI in Chrome.
6. Upload a small, text-based PDF.
7. Confirm that the backend extracts the text.
8. Confirm that the document is split into chunks.
9. Enter a question whose answer is present in the PDF.
10. Confirm that the actual configured SLM generates the answer.
11. Confirm that the answer is displayed in the UI.
12. Ask a second question without uploading the document again.
13. Test a question whose answer is not present in the document.
14. Test an invalid PDF and a backend connection failure.
15. Reset the document and verify that the UI returns to its initial state.

Do not report a test as successful unless it has actually been executed.

If you cannot execute the application, clearly distinguish generated code from tested functionality and provide instructions for me to verify it.

## 11. Instructions for Your Implementation

Before generating the code, determine whether you have access to my existing Google Colab notebook or repository.

If you have access:
- Inspect the existing implementation.
- Reuse the existing PDF extraction code.
- Reuse the configured base model and LoRA adapter paths.
- Reuse existing prompt construction and inference logic when appropriate.
- Avoid duplicating working functionality.

If you do not have access:
- Create a modular implementation.
- Clearly mark model identifiers, adapter paths and other unknown values as configuration requirements.
- Do not invent details about my existing notebook.
- List the information you need from me.

Prioritise a minimal, working end-to-end application over unnecessary features.

First, briefly explain the proposed architecture and identify any missing information needed to run the model. Then generate the actual files and provide exact setup instructions.

---

### 🤖 Assistant

> 🔧 called: read_file, execute_command

I'll start by checking what changed externally in `requirements.txt` and whether Node is available for the frontend build.

> 🔧 **read_file** — slm-document-analysis-backend/requirements.txt

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "node --version; npm --version; which cloudflared; uname -m"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && .venv/bin/pip install -q nodejs-wheel-binaries 2>&1 | tail -2; .venv/bin/node --version; .venv/bin/npm --version; .venv/bin/npm view vite version; .venv/bin/npm view react version; .venv/bin/npm view typescript version; .venv/bin/npm view @vitejs/plugin-react version; .venv/bin/npm view @types/react version; .venv/bin/npm view @types/react-dom version"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && ls .venv/bin | grep -iE 'node|npm|npx'; .venv/bin/python -c 'import nodejs_wheel,os;print(os.path.dirname(nodejs_wheel.__file__))'"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && ls .venv/lib/python3.14/site-packages/nodejs_wheel/bin; .venv/bin/python -m nodejs_wheel.executable node --version 2>&1 | head -2; .venv/bin/python -m nodejs_wheel npm --version 2>&1 | head -2"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "export PATH=\$HOME/ARC/Project/slm-document-analysis-backend/.venv/lib/python3.14/site-packages/nodejs_wheel/bin:\$PATH; node --version; npm --version; for p in vite react react-dom typescript @vitejs/plugin-react @types/react @types/react-dom; do echo \$p \$(npm view \$p version); done"

### 🤖 Assistant

> 🔧 called: write_file, write_file

Inline quoting through PowerShell keeps breaking, so I'll use a small helper script instead and delete it afterwards.

> 🔧 **write_file** — #!/usr/bin/env bash
# Temporary helper (deleted after use): puts a pip-installed Node on PATH, runs the given command.
export PATH="$HOME/ARC/Project/slm-document-analysis-backend/.venv/lib/python3.14/site-packages/nodejs_wheel/bin:$PATH"
exec "$@"


> 🔧 **write_file** — #!/usr/bin/env bash
node --version
npm --version
for p in vite react react-dom typescript @vitejs/plugin-react @types/react @types/react-dom; do
  echo "$p $(npm view $p version)"
done


### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_node.sh bash /home/ardraj/ARC/Project/.tmp_versions.sh

### 🤖 Assistant

> 🔧 called: write_file

The wheel's `npm` shim has a broken relative path, so I'll call npm's real entry script directly with node.

> 🔧 **write_file** — #!/usr/bin/env bash
# Temporary helper (deleted after use).
W="$HOME/ARC/Project/slm-document-analysis-backend/.venv/lib/python3.14/site-packages/nodejs_wheel"
ls "$W" "$W/lib/node_modules" 2>&1 | head -20
NPM="$W/lib/node_modules/npm/bin/npm-cli.js"
[ -f "$NPM" ] || NPM="$(find "$W" -name npm-cli.js | head -1)"
echo "npm-cli: $NPM"
export PATH="$W/bin:$PATH"
for p in vite react react-dom typescript @vitejs/plugin-react @types/react @types/react-dom; do
  echo "$p $(node "$NPM" view $p version 2>&1 | tail -1)"
done


### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_versions.sh

### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — #!/usr/bin/env bash
W="$HOME/ARC/Project/slm-document-analysis-backend/.venv/lib/python3.14/site-packages/nodejs_wheel"
export PATH="$W/bin:$PATH"
node "$W/lib/node_modules/npm/bin/npm-cli.js" view vite version 2>/dev/null
node "$W/lib/node_modules/npm/bin/npm-cli.js" view vite engines 2>/dev/null
node "$W/lib/node_modules/npm/bin/npm-cli.js" view @vitejs/plugin-react peerDependencies 2>/dev/null
node "$W/lib/node_modules/npm/bin/npm-cli.js" view typescript@7.0.2 bin 2>/dev/null


### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_versions.sh

### 🤖 Assistant

> 🔧 called: update_todo_list, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file



> 🔧 **update_todo_list** — [x] Inspect environment (no access to Mtech notebook/model config; Node via pip wheel in WSL)
[-] Scaffold React+TS+Vite frontend (config, types, api client, utils, components, styles)
[ ] Install deps, typecheck, build, unit tests for frontend
[ ] Create requirements-colab.txt + notebook builder script
[ ] Generate slm_document_analysis_backend.ipynb
[ ] Verify notebook locally (run its cells with tiny model) + run frontend API client e2e against live server
[ ] Root README.md (setup, API, Colab, e2e test procedure)
[ ] Clean temp helper files, final summary

> 🔧 **write_file** — {
  "name": "slm-document-analysis-frontend",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "typecheck": "tsc --noEmit",
    "build": "tsc --noEmit && vite build",
    "preview": "vite preview",
    "test": "vitest run"
  }
}


> 🔧 **write_file** — {
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "verbatimModuleSyntax": true,
    "erasableSyntaxOnly": true,
    "allowImportingTsExtensions": true,
    "noEmit": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "types": ["vite/client", "node"]
  },
  "include": ["src", "scripts", "vite.config.ts"]
}


> 🔧 **write_file** — import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The dev server must stay on port 5173: the backend's CORS_ORIGINS allows exactly this origin.
export default defineConfig({
  plugins: [react()],
  server: { host: "localhost", port: 5173, strictPort: true },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});


> 🔧 **write_file** — <!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>SLM Document Analysis</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>


> 🔧 **write_file** — # Copy this file to ".env" (in this folder) and edit it. ".env" is git-ignored.
# Vite only exposes variables that start with VITE_ and ships them to the browser,
# so NEVER put passwords, API keys or tunnel tokens here. The tunnel URL is not a secret
# credential, but anyone who knows it can call your backend: stop the tunnel when you are done.

# Base URL of the backend API, WITHOUT a trailing slash.
#   Colab:  the https://xxxx.trycloudflare.com URL printed by the notebook's tunnel cell
#   Local:  http://127.0.0.1:8000  (if you run slm-document-analysis-backend on the Mac)
VITE_API_BASE_URL=http://127.0.0.1:8000

# Client-side upload limit in MB (the backend enforces its own limit: MAX_UPLOAD_MB).
VITE_MAX_UPLOAD_MB=20


> 🔧 **write_file** — /// <reference types="vite/client" />


> 🔧 **write_file** — /** Central frontend configuration. The backend URL is set ONLY here (via VITE_API_BASE_URL). */

const rawBase = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export const API_BASE_URL: string = rawBase.trim().replace(/\/+$/, "");

const rawMax = Number(import.meta.env.VITE_MAX_UPLOAD_MB);
export const MAX_UPLOAD_MB: number = Number.isFinite(rawMax) && rawMax > 0 ? rawMax : 20;

export const MAX_QUESTION_LENGTH = 2000; // must match the backend's QuestionRequest limit


> 🔧 **write_file** — /** Response shapes returned by the backend (see README "API reference"). */

export interface HealthResponse {
  status: string;
  model_ready: boolean;
  model_name?: string | null;
  adapter_loaded?: boolean;
  device?: string | null;
  model_error?: string | null;
  documents_in_memory?: number;
}

export interface DocumentUploadResponse {
  document_id: string;
  filename: string;
  status: string;
  page_count: number;
  character_count: number;
  word_count: number;
  chunk_count: number;
  empty_page_numbers: number[];
  text_preview: string;
  text_preview_truncated: boolean;
}

export interface SupportingPassage {
  text: string;
  page_number: number;
  chunk_id: string;
  /** BM25 ranking score. It is NOT a probability or a confidence value. */
  retrieval_score: number;
}

export interface QuestionResponse {
  document_id: string;
  question: string;
  answer: string | null;
  answer_generated: boolean;
  message?: string | null;
  model_name?: string | null;
  inference_time_seconds?: number | null;
  prompt_tokens?: number | null;
  generated_tokens?: number | null;
  supporting_passages: SupportingPassage[];
  disclaimer?: string;
}

export interface DeleteResponse {
  document_id: string;
  deleted: boolean;
  message: string;
}


> 🔧 **write_file** — /**
 * Backend API client. All network access goes through createApi(), which takes the base URL
 * as a parameter (the app passes API_BASE_URL from config.ts). Every failure is converted into
 * an ApiRequestError with a message that is safe to show to the user.
 */

import type {
  DeleteResponse,
  DocumentUploadResponse,
  HealthResponse,
  QuestionResponse,
} from "./types.ts";

export type ErrorKind = "network" | "timeout" | "http" | "invalid_response";

export class ApiRequestError extends Error {
  kind: ErrorKind;
  status?: number;
  code?: string;

  constructor(message: string, kind: ErrorKind, status?: number, code?: string) {
    super(message);
    this.name = "ApiRequestError";
    this.kind = kind;
    this.status = status;
    this.code = code;
  }
}

type Guard<T> = (value: unknown) => value is T;

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

const isHealth: Guard<HealthResponse> = (v): v is HealthResponse =>
  isObject(v) && typeof v.status === "string" && typeof v.model_ready === "boolean";

const isUpload: Guard<DocumentUploadResponse> = (v): v is DocumentUploadResponse =>
  isObject(v) &&
  typeof v.document_id === "string" &&
  typeof v.filename === "string" &&
  typeof v.page_count === "number" &&
  typeof v.character_count === "number" &&
  typeof v.word_count === "number" &&
  typeof v.chunk_count === "number" &&
  typeof v.text_preview === "string";

const isQuestion: Guard<QuestionResponse> = (v): v is QuestionResponse =>
  isObject(v) &&
  typeof v.document_id === "string" &&
  typeof v.answer_generated === "boolean" &&
  (v.answer === null || typeof v.answer === "string") &&
  Array.isArray(v.supporting_passages);

const isDelete: Guard<DeleteResponse> = (v): v is DeleteResponse =>
  isObject(v) && typeof v.document_id === "string" && typeof v.deleted === "boolean";

function extractBackendError(body: unknown): { code: string; message: string } | null {
  if (isObject(body) && isObject(body.error)) {
    const { code, message } = body.error;
    if (typeof code === "string" && typeof message === "string") return { code, message };
  }
  return null;
}

export const TIMEOUTS_MS = { health: 10_000, upload: 120_000, question: 150_000, delete: 15_000 };

export function createApi(baseUrl: string, fetchImpl: typeof fetch = (...args) => fetch(...args)) {
  async function request<T>(
    path: string,
    init: RequestInit,
    timeoutMs: number,
    isValid: Guard<T>,
  ): Promise<T> {
    let response: Response;
    try {
      response = await fetchImpl(`${baseUrl}${path}`, {
        ...init,
        signal: AbortSignal.timeout(timeoutMs),
      });
    } catch (err) {
      if ((err as { name?: string }).name === "TimeoutError") {
        throw new ApiRequestError(
          "The request timed out. The backend may be busy, or the Colab session may have stopped.",
          "timeout",
        );
      }
      throw new ApiRequestError(
        "Cannot reach the backend. Check that the Colab notebook is running, the tunnel is active, " +
          "and VITE_API_BASE_URL is correct.",
        "network",
      );
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = undefined;
    }

    if (!response.ok) {
      const backendError = extractBackendError(body);
      if (backendError) {
        throw new ApiRequestError(backendError.message, "http", response.status, backendError.code);
      }
      throw new ApiRequestError(
        `The backend returned an unexpected response (HTTP ${response.status}). ` +
          "The Colab session or the tunnel may have stopped.",
        "invalid_response",
        response.status,
      );
    }
    if (!isValid(body)) {
      throw new ApiRequestError(
        "The backend returned a response in an unexpected format.",
        "invalid_response",
        response.status,
      );
    }
    return body;
  }

  return {
    health: () => request("/health", { method: "GET" }, TIMEOUTS_MS.health, isHealth),

    uploadDocument: (file: File) => {
      const form = new FormData();
      form.append("file", file); // the backend expects the field name "file"
      // Do not set Content-Type: the browser adds the multipart boundary itself.
      return request("/documents", { method: "POST", body: form }, TIMEOUTS_MS.upload, isUpload);
    },

    askQuestion: (documentId: string, question: string) =>
      request(
        `/documents/${encodeURIComponent(documentId)}/questions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question }),
        },
        TIMEOUTS_MS.question,
        isQuestion,
      ),

    deleteDocument: (documentId: string) =>
      request(
        `/documents/${encodeURIComponent(documentId)}`,
        { method: "DELETE" },
        TIMEOUTS_MS.delete,
        isDelete,
      ),
  };
}

export type Api = ReturnType<typeof createApi>;


> 🔧 **write_file** — /** Small pure helpers (unit-tested in utils.test.ts). */

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/** Returns a user-facing error message, or null when the file is acceptable. */
export function validatePdfFile(file: { name: string; size: number }, maxMb: number): string | null {
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    return "Unsupported file type. Please select a PDF file (.pdf).";
  }
  if (file.size === 0) {
    return "This file is empty (0 bytes). Please select a different PDF.";
  }
  if (file.size > maxMb * 1024 * 1024) {
    return `This file is ${formatBytes(file.size)}, which exceeds the ${maxMb} MB limit.`;
  }
  return null;
}

export function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}


> 🔧 **write_file** — import { describe, expect, it } from "vitest";
import { createApi, ApiRequestError } from "./api.ts";

const BASE = "http://backend.test";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const upload = {
  document_id: "a".repeat(32),
  filename: "x.pdf",
  status: "processed",
  page_count: 1,
  character_count: 10,
  word_count: 2,
  chunk_count: 1,
  empty_page_numbers: [],
  text_preview: "hello world",
  text_preview_truncated: false,
};

async function failureOf(p: Promise<unknown>): Promise<ApiRequestError> {
  try {
    await p;
  } catch (e) {
    if (e instanceof ApiRequestError) return e;
    throw e;
  }
  throw new Error("expected the request to fail");
}

describe("api client", () => {
  it("parses a valid health response and calls the right URL", async () => {
    let seen = "";
    const api = createApi(BASE, async (url) => {
      seen = String(url);
      return jsonResponse({ status: "ok", model_ready: true });
    });
    expect((await api.health()).model_ready).toBe(true);
    expect(seen).toBe(`${BASE}/health`);
  });

  it("sends the PDF as multipart field 'file' without a manual Content-Type", async () => {
    let init: RequestInit | undefined;
    const api = createApi(BASE, async (_url, i) => {
      init = i;
      return jsonResponse(upload, 201);
    });
    const file = new File([new Uint8Array([37, 80, 68, 70])], "x.pdf", { type: "application/pdf" });
    const result = await api.uploadDocument(file);
    expect(result.document_id).toBe(upload.document_id);
    expect(init?.method).toBe("POST");
    expect((init?.body as FormData).get("file")).toBeInstanceOf(File);
    expect(init?.headers).toBeUndefined();
  });

  it("posts the question as JSON to the document's URL", async () => {
    let url = "";
    let body = "";
    const api = createApi(BASE, async (u, i) => {
      url = String(u);
      body = String(i?.body);
      return jsonResponse({
        document_id: upload.document_id,
        question: "q",
        answer: "a",
        answer_generated: true,
        supporting_passages: [],
      });
    });
    const r = await api.askQuestion(upload.document_id, "What?");
    expect(r.answer).toBe("a");
    expect(url).toBe(`${BASE}/documents/${upload.document_id}/questions`);
    expect(JSON.parse(body)).toEqual({ question: "What?" });
  });

  it("reports a network failure clearly", async () => {
    const api = createApi(BASE, async () => {
      throw new TypeError("Failed to fetch");
    });
    const err = await failureOf(api.health());
    expect(err.kind).toBe("network");
    expect(err.message).toMatch(/Cannot reach the backend/);
  });

  it("reports a timeout", async () => {
    const api = createApi(BASE, async () => {
      throw new DOMException("timed out", "TimeoutError");
    });
    expect((await failureOf(api.health())).kind).toBe("timeout");
  });

  it("surfaces the backend's structured error message and code", async () => {
    const api = createApi(BASE, async () =>
      jsonResponse({ error: { code: "malformed_pdf", message: "The PDF is corrupt." } }, 422),
    );
    const err = await failureOf(api.uploadDocument(new File(["x"], "x.pdf")));
    expect(err).toMatchObject({ kind: "http", status: 422, code: "malformed_pdf", message: "The PDF is corrupt." });
  });

  it("handles non-JSON error pages (e.g. a stopped tunnel)", async () => {
    const api = createApi(BASE, async () => new Response("<html>Bad gateway</html>", { status: 502 }));
    const err = await failureOf(api.health());
    expect(err.kind).toBe("invalid_response");
    expect(err.message).toMatch(/HTTP 502/);
  });

  it("rejects a 200 response with an unexpected shape", async () => {
    const api = createApi(BASE, async () => jsonResponse({ hello: "world" }));
    expect((await failureOf(api.health())).kind).toBe("invalid_response");
  });
});


> 🔧 **write_file** — import { describe, expect, it } from "vitest";
import { formatBytes, validatePdfFile } from "./utils.ts";

describe("validatePdfFile", () => {
  it("accepts a normal PDF (case-insensitive extension)", () => {
    expect(validatePdfFile({ name: "Report.PDF", size: 1000 }, 20)).toBeNull();
  });
  it("rejects non-PDF files", () => {
    expect(validatePdfFile({ name: "notes.txt", size: 1000 }, 20)).toMatch(/Unsupported file type/);
    expect(validatePdfFile({ name: "pdf", size: 1000 }, 20)).toMatch(/Unsupported file type/);
  });
  it("rejects empty files", () => {
    expect(validatePdfFile({ name: "a.pdf", size: 0 }, 20)).toMatch(/empty/);
  });
  it("rejects oversized files", () => {
    expect(validatePdfFile({ name: "a.pdf", size: 21 * 1024 * 1024 }, 20)).toMatch(/exceeds the 20 MB limit/);
  });
});

describe("formatBytes", () => {
  it("formats sizes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.00 MB");
  });
});


### 🤖 Assistant

> 🔧 called: apply_diff, apply_diff, apply_diff

Next I'll add a full-text endpoint to the local backend. The UI needs it for the Expand and Copy Text buttons, because the upload response only carries a bounded preview.

> 🔧 **apply_diff** — slm-document-analysis-backend/app/config.py

> 🔧 **apply_diff** — slm-document-analysis-backend/app/schemas.py

> 🔧 **apply_diff** — slm-document-analysis-backend/app/main.py

### 🤖 Assistant

> 🔧 called: execute_command

The backend now has a `GET /documents/{id}/text` endpoint for the Expand and Copy Text buttons. I'll add it to the frontend API client and then build the UI.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/frontend && python3 - <<'EOF'
import re
p='src/types.ts'
s=open(p).read()
s+='''
export interface DocumentTextResponse {
  document_id: string;
  text: string;
  character_count: number;
  total_character_count: number;
  truncated: boolean;
}
'''
open(p,'w').write(s)
p='src/api.ts'
s=open(p).read()
s=s.replace('''  DocumentUploadResponse,
  HealthResponse,''','''  DocumentTextResponse,
  DocumentUploadResponse,
  HealthResponse,''',1)
s=s.replace('''const isQuestion:''','''const isText: Guard<DocumentTextResponse> = (v): v is DocumentTextResponse =>
  isObject(v) &&
  typeof v.document_id === \"string\" &&
  typeof v.text === \"string\" &&
  typeof v.total_character_count === \"number\" &&
  typeof v.truncated === \"boolean\";

const isQuestion:''',1)
s=s.replace('''    askQuestion:''','''    getDocumentText: (documentId: string) =>
      request(
        `/documents/${encodeURIComponent(documentId)}/text`,
        { method: \"GET\" },
        TIMEOUTS_MS.upload,
        isText,
      ),

    askQuestion:''',1)
open(p,'w').write(s)
EOF
grep -n 'isText\|getDocumentText' src/api.ts"

### 🤖 Assistant

> 🔧 called: apply_diff, insert_content

I'll use the file edit tools instead of shell heredocs.

> 🔧 **insert_content** — frontend/src/types.ts

> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:6
-------
import type {
  DeleteResponse,
  DocumentUploadResponse,
=======
import type {
  DeleteResponse,
  DocumentTextResponse,
  DocumentUploadResponse,
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:53
-------
const isQuestion: Guard<QuestionResponse> = (v): v is QuestionResponse =>
=======
const isText: Guard<DocumentTextResponse> = (v): v is DocumentTextResponse =>
  isObject(v) &&
  typeof v.document_id === "string" &&
  typeof v.text === "string" &&
  typeof v.total_character_count === "number" &&
  typeof v.truncated === "boolean";

const isQuestion: Guard<QuestionResponse> = (v): v is QuestionResponse =>
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:132
-------
    askQuestion: (documentId: string, question: string) =>
=======
    getDocumentText: (documentId: string) =>
      request(
        `/documents/${encodeURIComponent(documentId)}/text`,
        { method: "GET" },
        TIMEOUTS_MS.upload,
        isText,
      ),

    askQuestion: (documentId: string, question: string) =>
>>>>>>> REPLACE

### 🤖 Assistant

> 🔧 called: write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file, write_file

Now the React components and styles.

> 🔧 **write_file** — node_modules/
dist/
.env
.env.*
!.env.example
*.log
.DS_Store


> 🔧 **write_file** — import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiRequestError, createApi } from "./api.ts";
import { API_BASE_URL, MAX_UPLOAD_MB } from "./config.ts";
import { AnswerCard } from "./components/AnswerCard.tsx";
import { Header } from "./components/Header.tsx";
import { QuestionCard } from "./components/QuestionCard.tsx";
import { TextPreviewCard } from "./components/TextPreviewCard.tsx";
import { UploadCard } from "./components/UploadCard.tsx";
import type { DocumentUploadResponse, HealthResponse, QuestionResponse } from "./types.ts";

export type BackendState =
  | { kind: "checking" }
  | { kind: "online"; health: HealthResponse }
  | { kind: "offline"; message: string };

export type UploadState =
  | { kind: "idle" }
  | { kind: "uploading" }
  | { kind: "error"; message: string };

export default function App() {
  const api = useMemo(() => createApi(API_BASE_URL), []);

  const [backend, setBackend] = useState<BackendState>({ kind: "checking" });
  const [upload, setUpload] = useState<UploadState>({ kind: "idle" });
  const [document, setDocument] = useState<DocumentUploadResponse | null>(null);
  const [asking, setAsking] = useState(false);
  const [questionError, setQuestionError] = useState<string | null>(null);
  const [result, setResult] = useState<QuestionResponse | null>(null);
  const [resetNotice, setResetNotice] = useState<string | null>(null);
  // Bumped on reset so child components drop their local state (selected file, typed question...).
  const [resetCount, setResetCount] = useState(0);
  const askToken = useRef(0);

  const checkBackend = useCallback(async () => {
    setBackend({ kind: "checking" });
    try {
      setBackend({ kind: "online", health: await api.health() });
    } catch (err) {
      setBackend({ kind: "offline", message: errorMessage(err) });
    }
  }, [api]);

  useEffect(() => {
    void checkBackend();
  }, [checkBackend]);

  const handleUpload = async (file: File) => {
    setUpload({ kind: "uploading" });
    setResetNotice(null);
    try {
      const doc = await api.uploadDocument(file);
      setDocument(doc);
      setResult(null);
      setQuestionError(null);
      setUpload({ kind: "idle" });
    } catch (err) {
      setUpload({ kind: "error", message: errorMessage(err) });
    }
  };

  const handleAsk = async (question: string) => {
    if (!document) return;
    const token = ++askToken.current;
    setAsking(true);
    setQuestionError(null);
    try {
      const response = await api.askQuestion(document.document_id, question);
      if (token === askToken.current) setResult(response);
    } catch (err) {
      if (token === askToken.current) {
        setQuestionError(errorMessage(err));
        // If the backend lost the document (restart/expiry), the user must re-upload.
        if (err instanceof ApiRequestError && err.code === "document_not_found") {
          setDocument(null);
          setResult(null);
        }
      }
    } finally {
      if (token === askToken.current) setAsking(false);
    }
  };

  const handleReset = async () => {
    const current = document;
    askToken.current++; // ignore any in-flight answer
    setAsking(false);
    setDocument(null);
    setResult(null);
    setQuestionError(null);
    setUpload({ kind: "idle" });
    setResetCount((n) => n + 1);

    if (!current) {
      setResetNotice("Cleared.");
      return;
    }
    try {
      await api.deleteDocument(current.document_id);
      setResetNotice("Document removed from the browser and from the backend's in-memory storage.");
    } catch (err) {
      setResetNotice(
        `Cleared in the browser only. The backend could not confirm deletion (${errorMessage(err)}). ` +
          "In-memory documents expire automatically.",
      );
    }
  };

  const modelReady = backend.kind === "online" && backend.health.model_ready;

  return (
    <div className="page">
      <Header />
      <main className="container">
        <BackendBanner backend={backend} onRetry={checkBackend} />

        <UploadCard
          key={`upload-${resetCount}`}
          maxMb={MAX_UPLOAD_MB}
          uploading={upload.kind === "uploading"}
          error={upload.kind === "error" ? upload.message : null}
          processed={document}
          backendOffline={backend.kind === "offline"}
          onUpload={handleUpload}
        />

        {document && <TextPreviewCard api={api} doc={document} />}

        <QuestionCard
          key={`question-${resetCount}`}
          enabled={document !== null && modelReady}
          disabledReason={disabledReason(document !== null, backend)}
          asking={asking}
          error={questionError}
          onAsk={handleAsk}
        />

        {(result || asking) && <AnswerCard result={result} asking={asking} />}

        <section className="reset-row">
          <button type="button" className="btn btn-secondary" onClick={handleReset}>
            Start New Document / Reset
          </button>
          {resetNotice && <p className="muted small" role="status">{resetNotice}</p>}
        </section>

        <footer className="footer">
          Answers are generated by a small language model and may be incorrect. Uploading a document
          only processes it for question answering; it does not train or fine-tune the model.
        </footer>
      </main>
    </div>
  );
}

function errorMessage(err: unknown): string {
  return err instanceof ApiRequestError ? err.message : "An unexpected error occurred.";
}

function disabledReason(hasDocument: boolean, backend: BackendState): string | null {
  if (!hasDocument) return "Upload and process a document first.";
  if (backend.kind === "offline") return "The backend is unreachable.";
  if (backend.kind === "checking") return "Checking the backend...";
  if (!backend.health.model_ready) {
    return backend.health.model_error
      ? `The model is not ready: ${backend.health.model_error}`
      : "The model is not ready yet.";
  }
  return null;
}

function BackendBanner({ backend, onRetry }: { backend: BackendState; onRetry: () => void }) {
  if (backend.kind === "checking") {
    return <div className="banner banner-info" role="status">Checking backend connection...</div>;
  }
  if (backend.kind === "offline") {
    return (
      <div className="banner banner-error" role="alert">
        <div>
          <strong>Backend unavailable.</strong> {backend.message}
          <div className="small">Backend URL: <code>{API_BASE_URL}</code></div>
        </div>
        <button type="button" className="btn btn-small" onClick={onRetry}>Retry</button>
      </div>
    );
  }
  const { health } = backend;
  if (!health.model_ready) {
    return (
      <div className="banner banner-warn" role="alert">
        <div>
          <strong>Backend connected, but the model is not ready.</strong>{" "}
          {health.model_error ?? "Wait for model loading to finish in the Colab notebook."}
          <div className="small">You can upload documents, but questions are disabled until the model is ready.</div>
        </div>
        <button type="button" className="btn btn-small" onClick={onRetry}>Re-check</button>
      </div>
    );
  }
  return (
    <div className="banner banner-ok" role="status">
      Backend connected. Model ready: <strong>{health.model_name ?? "unknown"}</strong>
      {health.device ? ` (device: ${health.device})` : ""}
    </div>
  );
}


> 🔧 **write_file** — import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);


> 🔧 **write_file** — /** Copy text to the clipboard. Returns false if the browser refuses. */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    return false;
  }
}


> 🔧 **write_file** — :root {
  --navy: #1b2a49;
  --indigo: #3f51b5;
  --indigo-dark: #2f3e9e;
  --bg: #f4f6fa;
  --card: #ffffff;
  --border: #dfe3ec;
  --muted: #5f6b85;
  --ok: #1e7b4a;
  --warn: #8a5a00;
  --err: #b3261e;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #222b3e;
  line-height: 1.55;
}

* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); }
h1, h2, h3 { color: var(--navy); margin: 0; }
h2 { font-size: 1.2rem; margin-bottom: 1rem; }
h3 { font-size: 1.05rem; margin: 0 0 0.25rem; }
p { margin: 0.5rem 0; }
code { background: #eef1f8; padding: 0.05rem 0.35rem; border-radius: 4px; font-size: 0.85em; }

.container { max-width: 880px; margin: 0 auto; padding: 0 1.25rem; }
.header { background: var(--card); border-bottom: 1px solid var(--border); padding: 1.75rem 0 1.5rem; }
.title-row { display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; }
.header h1 { font-size: 1.9rem; }
.subtitle { color: var(--muted); max-width: 640px; margin: 0.5rem 0 0; }
.badge { background: #e8eaf8; color: var(--indigo-dark); border: 1px solid #c9cef0; font-size: 0.78rem;
  font-weight: 600; padding: 0.2rem 0.65rem; border-radius: 999px; }

main.container { padding-top: 1.5rem; padding-bottom: 3rem; display: flex; flex-direction: column; gap: 1.25rem; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1.5rem;
  box-shadow: 0 1px 2px rgba(27, 42, 73, 0.05); }
.card-stale { opacity: 0.65; }

.muted { color: var(--muted); }
.small { font-size: 0.85rem; }
.error-text { color: var(--err); font-weight: 500; }
.visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }

.btn { font: inherit; font-weight: 600; border-radius: 9px; padding: 0.6rem 1.15rem; cursor: pointer;
  border: 1px solid transparent; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: var(--indigo); color: #fff; }
.btn-primary:hover:not(:disabled) { background: var(--indigo-dark); }
.btn-secondary { background: #fff; color: var(--indigo-dark); border-color: #b9c0e6; }
.btn-secondary:hover:not(:disabled) { background: #f0f2fb; }
.btn-small { padding: 0.35rem 0.8rem; background: #fff; border-color: var(--border); }
.btn:focus-visible, .chip:focus-visible, textarea:focus-visible, .text-box:focus-visible {
  outline: 3px solid #9fa8ea; outline-offset: 2px; }
.button-row { display: flex; gap: 0.75rem; align-items: center; flex-wrap: wrap; margin-top: 1rem; }

.dropzone { border: 2px dashed #b9c0e6; background: #f8f9fe; border-radius: 12px; padding: 1.5rem;
  text-align: center; margin: 1rem 0; }
.dropzone-active { background: #eceffd; border-color: var(--indigo); }
.dropzone-icon { font-size: 2rem; }
.file-info { margin: 0.5rem 0 1rem; }

.notice, .banner { border-radius: 10px; padding: 0.75rem 1rem; border: 1px solid; margin: 0.75rem 0; font-size: 0.93rem; }
.notice-warn, .banner-warn { background: #fff7e6; border-color: #f0d9a3; color: var(--warn); }
.notice-error, .banner-error { background: #fdecea; border-color: #f2bcb7; color: var(--err); }
.notice-ok, .banner-ok { background: #e8f6ee; border-color: #b4dfc6; color: var(--ok); }
.banner-info { background: #eef1fb; border-color: #cfd6f2; color: var(--navy); }
.banner { margin: 0; display: flex; justify-content: space-between; align-items: center; gap: 1rem; }

.status-row { display: flex; align-items: center; gap: 0.6rem; margin-top: 1rem; color: var(--navy); }
.spinner { width: 18px; height: 18px; border: 3px solid #cfd6f2; border-top-color: var(--indigo);
  border-radius: 50%; animation: spin 0.9s linear infinite; flex: none; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .spinner { animation-duration: 3s; } }

.stats, .meta { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0 0 1rem; }
.stats > div, .meta > div { background: #f4f6fb; border: 1px solid var(--border); border-radius: 10px;
  padding: 0.5rem 0.9rem; min-width: 110px; }
dt { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
dd { margin: 0; font-weight: 600; color: var(--navy); }

.text-box { background: #f8f9fc; border: 1px solid var(--border); border-radius: 10px; padding: 0.9rem;
  max-height: 160px; overflow: auto; white-space: pre-wrap; word-break: break-word; margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 0.85rem; }
.text-box-expanded { max-height: 480px; }

textarea { width: 100%; font: inherit; padding: 0.75rem; border: 1px solid #c4cada; border-radius: 10px; resize: vertical; }
textarea:disabled { background: #f1f3f8; }
.chips { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin-top: 0.75rem; }
.chip { font: inherit; font-size: 0.85rem; background: #eef1fb; color: var(--indigo-dark); border: 1px solid #cfd6f2;
  border-radius: 999px; padding: 0.25rem 0.8rem; cursor: pointer; }
.chip:disabled { opacity: 0.5; cursor: not-allowed; }

.label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
.qa-block { margin-bottom: 1rem; }
.answer { white-space: pre-wrap; font-size: 1.02rem; background: #f4f6fb; border-left: 4px solid var(--indigo);
  padding: 0.8rem 1rem; border-radius: 6px; }

.evidence { margin-top: 1.25rem; border-top: 1px solid var(--border); padding-top: 1rem; }
.passages { padding-left: 1.2rem; margin: 0.75rem 0 0; }
.passage { margin-bottom: 0.9rem; }
.passage blockquote { margin: 0.3rem 0 0; padding: 0.6rem 0.9rem; background: #fafbfd; border: 1px solid var(--border);
  border-radius: 8px; white-space: pre-wrap; word-break: break-word; font-size: 0.9rem; }

.reset-row { display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; }
.footer { color: var(--muted); font-size: 0.82rem; border-top: 1px solid var(--border); padding-top: 1rem; }

@media (max-width: 640px) {
  .banner { flex-direction: column; align-items: flex-start; }
  .header h1 { font-size: 1.5rem; }
}


> 🔧 **write_file** — export function Header() {
  return (
    <header className="header">
      <div className="container header-inner">
        <div className="title-row">
          <h1>SLM Document Analysis</h1>
          <span className="badge">Proof of Concept</span>
        </div>
        <p className="subtitle">
          Upload a document, ask questions, and explore how Small Language Models can understand
          document content.
        </p>
      </div>
    </header>
  );
}


> 🔧 **write_file** — import { useRef, useState } from "react";
import type { DocumentUploadResponse } from "../types.ts";
import { formatBytes, formatNumber, validatePdfFile } from "../utils.ts";

interface Props {
  maxMb: number;
  uploading: boolean;
  error: string | null;
  processed: DocumentUploadResponse | null;
  backendOffline: boolean;
  onUpload: (file: File) => void;
}

export function UploadCard({ maxMb, uploading, error, processed, backendOffline, onUpload }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = (candidate: File | undefined) => {
    if (!candidate) return;
    const problem = validatePdfFile(candidate, maxMb);
    if (problem) {
      setFile(null);
      setFileError(problem);
      return;
    }
    setFile(candidate);
    setFileError(null);
  };

  return (
    <section className="card" aria-labelledby="step1">
      <h2 id="step1">Step 1: Upload Your Document</h2>

      <div className="notice notice-warn">
        <strong>Text-based PDFs only.</strong> This version reads the text already embedded in a PDF.
        Scanned or image-only PDFs have no extractable text and will be rejected; OCR is not implemented.
      </div>

      <div
        className={`dropzone${dragging ? " dropzone-active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          choose(e.dataTransfer.files[0]);
        }}
      >
        <div className="dropzone-icon" aria-hidden="true">&#128196;</div>
        <p><strong>Drag and drop a PDF here</strong></p>
        <p className="muted small">or</p>
        <button type="button" className="btn btn-secondary" onClick={() => inputRef.current?.click()} disabled={uploading}>
          Browse Files
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(e) => {
            choose(e.target.files?.[0]);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
        <p className="muted small">PDF only, up to {maxMb} MB</p>
      </div>

      {file && (
        <p className="file-info">
          <span aria-hidden="true">&#128206;</span> <strong>{file.name}</strong>
          <span className="muted"> &middot; {formatBytes(file.size)}</span>
        </p>
      )}
      {fileError && <p className="error-text" role="alert">{fileError}</p>}

      <button
        type="button"
        className="btn btn-primary"
        disabled={!file || uploading || backendOffline}
        onClick={() => file && onUpload(file)}
      >
        {uploading ? "Processing..." : "Upload and Process Document"}
      </button>
      {backendOffline && <span className="muted small"> The backend is unreachable.</span>}

      {uploading && (
        <div className="status-row" role="status">
          <span className="spinner" aria-hidden="true" />
          Uploading, extracting text and splitting it into chunks...
        </div>
      )}
      {error && <div className="notice notice-error" role="alert"><strong>Processing failed.</strong> {error}</div>}
      {processed && !uploading && (
        <div className="notice notice-ok" role="status">
          <strong>Processing complete.</strong> <em>{processed.filename}</em>: {formatNumber(processed.page_count)}{" "}
          page(s), {formatNumber(processed.chunk_count)} chunk(s). You can now ask questions.
          {processed.empty_page_numbers.length > 0 && (
            <div className="small">
              No text found on page(s): {processed.empty_page_numbers.join(", ")}.
            </div>
          )}
        </div>
      )}
    </section>
  );
}


> 🔧 **write_file** — import { useState } from "react";
import type { Api } from "../api.ts";
import { ApiRequestError } from "../api.ts";
import type { DocumentUploadResponse } from "../types.ts";
import { formatNumber } from "../utils.ts";
import { copyToClipboard } from "../clipboard.ts";

interface Props {
  api: Api;
  doc: DocumentUploadResponse;
}

type FullText = { text: string; truncated: boolean; total: number };

export function TextPreviewCard({ api, doc }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [full, setFull] = useState<FullText | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  // Fetch the complete text only when needed (expand / copy); the upload response carries a preview.
  const loadFull = async (): Promise<FullText | null> => {
    if (full) return full;
    setLoading(true);
    setError(null);
    try {
      const r = await api.getDocumentText(doc.document_id);
      const loaded = { text: r.text, truncated: r.truncated, total: r.total_character_count };
      setFull(loaded);
      return loaded;
    } catch (err) {
      setError(err instanceof ApiRequestError ? err.message : "Could not load the full text.");
      return null;
    } finally {
      setLoading(false);
    }
  };

  const toggle = async () => {
    if (!expanded) await loadFull();
    setExpanded((v) => !v);
  };

  const copy = async () => {
    const loaded = await loadFull();
    const text = loaded?.text ?? doc.text_preview;
    const ok = await copyToClipboard(text);
    setCopied(ok ? "Copied to clipboard." : "Copy failed. Select the text and copy it manually.");
    window.setTimeout(() => setCopied(null), 3000);
  };

  const shown = expanded && full ? full.text : doc.text_preview;

  return (
    <section className="card" aria-labelledby="step2">
      <h2 id="step2">Step 2: Extracted Document Text</h2>

      <dl className="stats">
        <div><dt>Characters</dt><dd>{formatNumber(doc.character_count)}</dd></div>
        <div><dt>Words</dt><dd>{formatNumber(doc.word_count)}</dd></div>
        <div><dt>Pages</dt><dd>{formatNumber(doc.page_count)}</dd></div>
        <div><dt>Chunks created</dt><dd>{formatNumber(doc.chunk_count)}</dd></div>
      </dl>

      <pre className={`text-box${expanded ? " text-box-expanded" : ""}`} tabIndex={0}>{shown}</pre>
      {!expanded && doc.text_preview_truncated && (
        <p className="muted small">Showing a preview of the extracted text. Use "Expand" to view more.</p>
      )}
      {expanded && full?.truncated && (
        <p className="muted small">
          Showing the first {formatNumber(full.text.length)} of {formatNumber(full.total)} characters.
          Question answering still uses the whole document.
        </p>
      )}

      <div className="button-row">
        <button type="button" className="btn btn-secondary" onClick={toggle} disabled={loading}>
          {loading ? "Loading..." : expanded ? "Collapse" : "Expand"}
        </button>
        <button type="button" className="btn btn-secondary" onClick={copy} disabled={loading}>
          Copy Text
        </button>
        {copied && <span className="muted small" role="status">{copied}</span>}
      </div>
      {error && <p className="error-text" role="alert">{error}</p>}
    </section>
  );
}


> 🔧 **write_file** — import { useState } from "react";
import { MAX_QUESTION_LENGTH } from "../config.ts";

// Suggestions only: clicking one fills the input. No answers are attached to them.
const EXAMPLE_QUESTIONS = [
  "What is the main topic of this document?",
  "What are the key conclusions or findings?",
  "Who is mentioned in the document?",
];

interface Props {
  enabled: boolean;
  disabledReason: string | null;
  asking: boolean;
  error: string | null;
  onAsk: (question: string) => void;
}

export function QuestionCard({ enabled, disabledReason, asking, error, onAsk }: Props) {
  const [question, setQuestion] = useState("");
  const trimmed = question.trim();
  const canSubmit = enabled && !asking && trimmed.length > 0 && trimmed.length <= MAX_QUESTION_LENGTH;

  return (
    <section className="card" aria-labelledby="step3">
      <h2 id="step3">Step 3: Ask a Question</h2>

      <label htmlFor="question" className="visually-hidden">Your question</label>
      <textarea
        id="question"
        rows={4}
        value={question}
        maxLength={MAX_QUESTION_LENGTH}
        placeholder="Ask a question about your uploaded document..."
        disabled={!enabled || asking}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSubmit) onAsk(trimmed);
        }}
      />

      <div className="chips" aria-label="Example questions">
        <span className="muted small">Try:</span>
        {EXAMPLE_QUESTIONS.map((q) => (
          <button key={q} type="button" className="chip" disabled={!enabled || asking} onClick={() => setQuestion(q)}>
            {q}
          </button>
        ))}
      </div>

      <div className="button-row">
        <button type="button" className="btn btn-primary" disabled={!canSubmit} onClick={() => onAsk(trimmed)}>
          {asking ? "Generating..." : "Generate Answer"}
        </button>
        {!enabled && disabledReason && <span className="muted small">{disabledReason}</span>}
        {enabled && <span className="muted small">Tip: Cmd/Ctrl + Enter to submit</span>}
      </div>

      {asking && (
        <div className="status-row" role="status">
          <span className="spinner" aria-hidden="true" />
          Retrieving relevant passages and generating an answer. This can take a while on a small GPU or CPU...
        </div>
      )}
      {error && <div className="notice notice-error" role="alert"><strong>Could not get an answer.</strong> {error}</div>}
    </section>
  );
}


> 🔧 **write_file** — import { useState } from "react";
import type { QuestionResponse } from "../types.ts";
import { copyToClipboard } from "../clipboard.ts";

interface Props {
  result: QuestionResponse | null;
  asking: boolean;
}

export function AnswerCard({ result, asking }: Props) {
  const [copied, setCopied] = useState<string | null>(null);

  if (!result) {
    return (
      <section className="card" aria-labelledby="step4">
        <h2 id="step4">Step 4: Generated Answer</h2>
        <p className="muted">Waiting for the model...</p>
      </section>
    );
  }

  const copy = async () => {
    if (!result.answer) return;
    const ok = await copyToClipboard(result.answer);
    setCopied(ok ? "Copied to clipboard." : "Copy failed.");
    window.setTimeout(() => setCopied(null), 3000);
  };

  return (
    <section className={`card${asking ? " card-stale" : ""}`} aria-labelledby="step4">
      <h2 id="step4">Step 4: Generated Answer</h2>
      {asking && <p className="muted small">Showing the previous answer while a new one is generated...</p>}

      <div className="qa-block">
        <div className="label">Question</div>
        <p>{result.question}</p>
      </div>

      {result.answer_generated && result.answer ? (
        <div className="qa-block">
          <div className="label">Answer</div>
          <p className="answer">{result.answer}</p>
        </div>
      ) : (
        <div className="notice notice-warn" role="status">
          <strong>No answer was generated.</strong>{" "}
          {result.message ?? "No relevant passage was found in the document."}
        </div>
      )}

      <dl className="meta">
        {result.model_name && (<div><dt>Model</dt><dd>{result.model_name}</dd></div>)}
        {typeof result.inference_time_seconds === "number" && (
          <div><dt>Inference time</dt><dd>{result.inference_time_seconds.toFixed(2)} s</dd></div>
        )}
      </dl>

      <div className="button-row">
        <button type="button" className="btn btn-secondary" onClick={copy} disabled={!result.answer}>
          Copy Answer
        </button>
        {copied && <span className="muted small" role="status">{copied}</span>}
      </div>

      {result.disclaimer && <p className="muted small">{result.disclaimer}</p>}

      {result.supporting_passages.length > 0 && (
        <div className="evidence">
          <h3>Supporting Evidence from the Document</h3>
          <p className="muted small">
            Passages retrieved from the document and given to the model. The score is a keyword
            ranking value, not a confidence in the answer.
          </p>
          <ol className="passages">
            {result.supporting_passages.map((p) => (
              <li key={p.chunk_id} className="passage">
                <div className="passage-head small muted">
                  Page {p.page_number} &middot; <code>{p.chunk_id}</code> &middot; keyword score{" "}
                  {p.retrieval_score.toFixed(2)}
                </div>
                <blockquote>{p.text}</blockquote>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}


### 🤖 Assistant

> 🔧 called: write_file, execute_command

`.gitignore` can't be written through the file tool, so I'll rename `.gitignore.new` with a shell command. Then I'll install the frontend dependencies with a reusable npm helper script.

> 🔧 **write_file** — #!/usr/bin/env bash
# Temporary helper (deleted at the end): run npm using the pip-installed Node inside WSL.
W="$HOME/ARC/Project/slm-document-analysis-backend/.venv/lib/python3.14/site-packages/nodejs_wheel"
export PATH="$W/bin:$PATH"
cd "$HOME/ARC/Project/frontend" || exit 1
if [ "$1" = "npm" ]; then shift; exec node "$W/lib/node_modules/npm/bin/npm-cli.js" "$@"; fi
exec "$@"


> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "mv /home/ardraj/ARC/Project/frontend/.gitignore.new /home/ardraj/ARC/Project/frontend/.gitignore && bash /home/ardraj/ARC/Project/.tmp_node.sh npm install react react-dom 2>&1 | tail -3 && bash /home/ardraj/ARC/Project/.tmp_node.sh npm install -D vite @vitejs/plugin-react typescript @types/react @types/react-dom @types/node vitest 2>&1 | tail -5; cat /home/ardraj/ARC/Project/frontend/package.json"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "bash /home/ardraj/ARC/Project/.tmp_node.sh npm run build 2>&1 | tail -25; bash /home/ardraj/ARC/Project/.tmp_node.sh npm test 2>&1 | tail -25"

### 🤖 Assistant

> 🔧 called: apply_diff



> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:12
-------
from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence

from app.config import Settings
from app.prompting import build_user_message
from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
=======
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

from app.config import Settings

if TYPE_CHECKING:  # type hints only; keeps model loading independent of retrieval code
    from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:293
-------
    def _encode(self, question: str, passages: Sequence[RetrievedChunk]) -> Any:
        user_message = build_user_message(question, passages)
=======
    def _encode(self, question: str, passages: Sequence[RetrievedChunk]) -> Any:
        from app.prompting import build_user_message

        user_message = build_user_message(question, passages)
>>>>>>> REPLACE

### 🤖 Assistant

> 🔧 called: apply_diff, update_todo_list

I made an unnecessary edit to `model_service.py` there. Since the notebook builder will inline these modules, I'm reverting it to the original tested version.

> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:18
-------
from typing import TYPE_CHECKING, Any, Optional, Protocol, Sequence

from app.config import Settings

if TYPE_CHECKING:  # type hints only; keeps model loading independent of retrieval code
    from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
=======
from typing import Any, Optional, Protocol, Sequence

from app.config import Settings
from app.prompting import build_user_message
from app.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:330
-------
    def _encode(self, question: str, passages: Sequence[RetrievedChunk]) -> Any:
        from app.prompting import build_user_message

        user_message = build_user_message(question, passages)
=======
    def _encode(self, question: str, passages: Sequence[RetrievedChunk]) -> Any:
        user_message = build_user_message(question, passages)
>>>>>>> REPLACE

> 🔧 **update_todo_list** — [x] Inspect environment (no access to Mtech notebook/model config; Node via pip wheel in WSL)
[x] Scaffold React+TS+Vite frontend, install deps, typecheck/build, 13 unit tests pass
[x] Add GET /documents/{id}/text to backend (for Expand/Copy Text)
[-] Create colab/ builder script (inlines tested app/*.py into notebook cells) + requirements-colab.txt
[ ] Generate slm_document_analysis_backend.ipynb
[ ] Verify notebook locally (execute its cells with tiny model) + backend pytest incl. new endpoint test
[ ] Run frontend e2e against live server (jsdom or headless browser if possible)
[ ] Root README.md (setup, API, Colab, e2e test procedure)
[ ] Clean temp helper files, final summary

### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — """Build ``slm_document_analysis_backend.ipynb`` (Google Colab) and ``requirements.txt``.

The notebook's Python is NOT written separately: it is assembled from the tested modules in
``../slm-document-analysis-backend/app/`` (with the ``from app.x import ...`` lines removed, since
everything lives in one notebook namespace). That keeps the notebook and the unit-tested
backend identical.

Usage (from this folder):  python build_notebook.py
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent / "slm-document-analysis-backend" / "app"

PIP_PACKAGES = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "python-multipart>=0.0.9",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "pymupdf>=1.24",
    "transformers>=4.56",
    "peft>=0.13",
]

_APP_IMPORT = re.compile(r"^from app\.\w+ import (?:\([^)]*\)|[^\n]*)\n", re.MULTILINE)
_FUTURE = re.compile(r"^from __future__ import annotations\n", re.MULTILINE)


def module_source(name: str) -> str:
    """Return a module's source, minus intra-package imports and the __future__ import."""
    src = (APP / f"{name}.py").read_text(encoding="utf-8")
    src = _APP_IMPORT.sub("", src)
    src = _FUTURE.sub("", src)
    return src.strip() + "\n"


def join_modules(*parts: str) -> str:
    return "from __future__ import annotations\n\n" + "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Cell contents
# --------------------------------------------------------------------------------------
INTRO_MD = """\
# SLM Document Analysis: Backend (Google Colab)

This notebook runs the **backend** for the *SLM Document Analysis* proof of concept:
PDF text extraction, chunking, retrieval, and question answering with your **Small Language Model
(base model + optional LoRA adapter)**. The React frontend on your MacBook talks to it over HTTPS
through a temporary tunnel.

**Inference only.** Uploading a PDF never trains or fine-tunes anything. The LoRA adapter is a
separate, already-trained artefact that is only *loaded* here.

**How to run**
1. `Runtime -> Change runtime type -> GPU` (a T4 is enough for models of up to a few billion parameters).
2. Fill in **Cell 2** (model and adapter). No model is assumed: you must supply it.
3. Run the cells **in order**, 1 to 11. Cell 11 prints the URL that goes into the frontend's `.env`.

**Limitations (read this)**
* Documents are stored **in memory only**. They are lost when the Colab session restarts or is
  disconnected, and nothing is written to disk.
* The tunnel URL changes every time you start it. Update the frontend `.env` and restart `npm run dev`.
* Text-based PDFs only. **There is no OCR**: scanned PDFs are rejected.
* Retrieval is lexical (BM25). It is **not** semantic search.
* Answers come from a small model and can be wrong.

> The Python in this notebook is generated from the unit-tested backend modules in
> `slm-document-analysis-backend/app/` by `colab/build_notebook.py`. If you change the backend
> source, rebuild the notebook instead of editing both.
"""

CELL1_MD = """\
## Cell 1: Install dependencies

Only what the code imports is installed. **PyTorch is not reinstalled**: Colab already ships a
CUDA build of it, and replacing it would be slow and could break the GPU runtime.

* `fastapi`, `uvicorn`, `python-multipart`: the API server and multipart (file upload) parsing
* `pydantic`, `pydantic-settings`: request validation and the settings object
* `pymupdf`: PDF text extraction
* `transformers`, `peft`: loading the base model and the LoRA adapter

If pip upgrades `transformers` or `pydantic` and a later cell raises an import error, use
`Runtime -> Restart session` and run the cells again from Cell 1.
"""

CELL1_CODE = r'''%pip install -q __PIP_ARGS__

import importlib.metadata as _md

for _pkg in ["fastapi", "uvicorn", "pydantic", "pydantic-settings", "pymupdf", "torch", "transformers", "peft"]:
    print(f"{_pkg:<18} {_md.version(_pkg)}")
'''

CELL2_MD = """\
## Cell 2: Configuration

Edit the values in the **"EDIT THESE VALUES"** block. Everything below it is the settings class
(taken from the tested backend) and normally needs no changes.

**Values you must supply**

| Variable | Meaning |
|---|---|
| `MODEL_NAME_OR_PATH` | The **same base model** you fine-tuned: a Hugging Face id (for example `"org/model-name"`) or a path such as `"/content/drive/MyDrive/.../base_model"`. |
| `ADAPTER_PATH` | Folder containing the saved LoRA adapter (it holds `adapter_config.json`). Leave `""` to run the base model alone. |
| `MOUNT_GOOGLE_DRIVE` | Set to `True` if the model or adapter lives in Google Drive. Cell 3 then mounts Drive. |

Nothing is pre-filled. If `MODEL_NAME_OR_PATH` is empty, Cell 3 stops with an explanatory error
instead of picking a model for you.

**Private or gated Hugging Face models:** store your token in Colab's **Secrets** panel (key icon in
the left sidebar) under the name `HF_TOKEN` and enable notebook access. Cell 3 reads it from there.
Never paste tokens into a cell.
"""

CELL2_USER_BLOCK = '''\
# ======================================================================================
#                                 EDIT THESE VALUES
# ======================================================================================

# --- Model (REQUIRED: nothing is assumed) -----------------------------------------------
MODEL_NAME_OR_PATH = ""      # <<< FILL IN: base model id or path (the one your adapter was trained on)
ADAPTER_PATH = ""            # <<< FILL IN if you use a LoRA adapter, e.g. "/content/drive/MyDrive/.../adapter"
TOKENIZER_NAME_OR_PATH = ""  # optional; leave empty to use the base model's tokenizer
MOUNT_GOOGLE_DRIVE = False   # True if the model/adapter is stored in Google Drive

# --- Device and precision -----------------------------------------------------------------
MODEL_DEVICE = "auto"        # "auto" (cuda if available), "cuda" or "cpu"
MODEL_DTYPE = "auto"         # "auto" (float16 on GPU, float32 on CPU), "float16", "bfloat16", "float32"
MERGE_ADAPTER = False        # True merges the LoRA weights into the base model (slightly faster inference)
TRUST_REMOTE_CODE = False    # only set True for models that require custom code you trust

# --- Chunking and retrieval -----------------------------------------------------------------
CHUNK_SIZE = 1000            # characters per chunk
CHUNK_OVERLAP = 150          # characters shared between consecutive chunks (must be < CHUNK_SIZE)
RETRIEVAL_TOP_K = 4          # maximum number of chunks given to the model
MAX_CONTEXT_CHARS = 6000     # maximum total characters of retrieved context

# --- Generation -----------------------------------------------------------------------------
MAX_NEW_TOKENS = 256         # maximum answer length in tokens
MAX_INPUT_TOKENS = 3072      # prompt budget (also capped by the model's context window)
# Keep this BELOW ~100 s: Cloudflare tunnels cut requests off at about 100 s with an HTML error page.
GENERATION_TIMEOUT_SECONDS = 90

# --- Uploads -------------------------------------------------------------------------------
MAX_UPLOAD_MB = 20

# --- API server and CORS ---------------------------------------------------------------------
API_HOST = "127.0.0.1"       # keep as is: the tunnel connects locally; never expose 0.0.0.0
API_PORT = 8000
# Origins of the FRONTEND (where the browser page is served from), NOT the tunnel URL.
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
START_TUNNEL = True          # False = start only the local server (no public URL)

# ======================================================================================
#                      END OF EDITABLE VALUES (settings class below)
# ======================================================================================
'''

CELL2_BUILD_SETTINGS = '''

settings = Settings(
    _env_file=None,  # ignore any .env file; the values above are the single source of truth
    model_name_or_path=MODEL_NAME_OR_PATH,
    adapter_path=ADAPTER_PATH,
    tokenizer_name_or_path=TOKENIZER_NAME_OR_PATH,
    model_device=MODEL_DEVICE,
    model_dtype=MODEL_DTYPE,
    merge_adapter=MERGE_ADAPTER,
    trust_remote_code=TRUST_REMOTE_CODE,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    retrieval_top_k=RETRIEVAL_TOP_K,
    max_context_chars=MAX_CONTEXT_CHARS,
    max_new_tokens=MAX_NEW_TOKENS,
    max_input_tokens=MAX_INPUT_TOKENS,
    generation_timeout_seconds=GENERATION_TIMEOUT_SECONDS,
    max_upload_mb=MAX_UPLOAD_MB,
    cors_origins=",".join(ALLOWED_ORIGINS),
    load_model_on_startup=False,  # the model is loaded explicitly in Cell 3
)


def get_settings() -> Settings:
    """Used by create_app() (Cell 9) when no settings object is passed in."""
    return settings


print("Configuration OK")
print("  model   :", settings.model_name_or_path or "<<< NOT SET: fill in Cell 2")
print("  adapter :", settings.adapter_path or "(none: base model only)")
print("  chunks  :", settings.chunk_size, "chars, overlap", settings.chunk_overlap)
print("  origins :", settings.cors_origin_list)
'''

CELL3_MD = """\
## Cell 3: Load the model

`ModelService` loads the tokenizer, the **base model**, and (if `ADAPTER_PATH` is set) the **LoRA adapter**
with `peft.PeftModel.from_pretrained`, then switches the model to evaluation (inference) mode.
It is loaded **once** and reused for every question.

* If `MOUNT_GOOGLE_DRIVE = True`, Drive is mounted first (Colab will ask you to authorise it).
* A Hugging Face token, if needed, is read from Colab Secrets (`HF_TOKEN`).
* Model loading is deliberately separate from the upload/extraction/question functions below.
* It never substitutes another model: a missing or wrong path stops with an error.

This class also contains the generation method (`generate_answer`) used by Cell 8 and the API.
Loading a few-billion-parameter model can take several minutes.
"""

CELL3_LOAD_CODE = '''\


# --------------------------------------------------------------------------------------
# Load the configured model (runs once)
# --------------------------------------------------------------------------------------
if MOUNT_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")

# A Hugging Face token is only needed for gated/private models. Read it from Colab Secrets.
try:
    from google.colab import userdata

    _hf_token = userdata.get("HF_TOKEN")
    if _hf_token:
        os.environ["HF_TOKEN"] = _hf_token
except Exception:  # not running on Colab, or no HF_TOKEN secret / access not granted
    pass

model_service = ModelService(settings)
try:
    model_service.load()
except ModelServiceError as exc:
    print("MODEL NOT LOADED:", exc.message)
    raise
print("Model ready:", model_service.status)
'''

CELL4_MD = """\
## Cell 4: PDF text extraction

`process_pdf(filename, data, max_bytes=...)` validates the upload and extracts text **page by page**
with PyMuPDF. Page numbers are 1-based and preserved.

It raises a specific error for: wrong file type (extension and `%PDF-` signature are both checked),
empty file, file larger than `MAX_UPLOAD_MB`, corrupt or password-protected PDF, extraction failure,
and PDFs with **no meaningful text**.

**No OCR.** A scanned (image-only) PDF is rejected with `NoExtractableTextError`; the code does not
pretend to read it.
"""

CELL5_MD = """\
## Cell 5: Text chunking and in-memory document storage

**Chunking** (`chunk_pages`): each page is split independently into windows of `CHUNK_SIZE`
characters that overlap by `CHUNK_OVERLAP`. Cuts are moved back to whitespace so words are not split,
and every chunk is an exact slice of the original page text (nothing is rewritten). The end of each
chunk strictly advances, so the loop always terminates and no chunk is a duplicate-only fragment.
Each chunk carries `chunk_id`, `document_id` and `page_number`.

**Storage** (`DocumentStore`): a thread-safe dictionary in this Python process.

> **In-memory storage is temporary.** It is not shared between processes and is **lost when the Colab
> session restarts or disconnects**. Old documents also expire after `DOCUMENT_TTL_MINUTES` (60) of
> inactivity, and the least recently used are evicted beyond `MAX_DOCUMENTS` (50).
"""

CELL6_MD = """\
## Cell 6: Relevant chunk selection (BM25)

**Method: BM25, a lexical (keyword) ranking function.** It needs no extra model, which keeps GPU
memory free for the SLM. **This is not semantic retrieval**: it matches words, not meaning, so a
question phrased very differently from the document can miss the right passage.

* A retriever is built **per document** from that document's chunks only, so it cannot return chunks
  of another document.
* At most `RETRIEVAL_TOP_K` chunks and `MAX_CONTEXT_CHARS` characters are selected.
* If no chunk shares any (non stop-word) term with the question, the result is **empty** and the API
  answers "no relevant passage found" without calling the model.
* The `Retriever` protocol is the replacement point for an embedding-based retriever later.
"""

CELL7_MD = """\
## Cell 7: Prompt construction

`build_user_message(question, passages)` produces the instruction, the numbered passages (with page
numbers) and the question. Cell 3's `ModelService` then wraps this in the **model's own chat
template** (`tokenizer.apply_chat_template`) when the tokenizer has one, and uses a plain
`Answer:` format otherwise. Lowest-ranked passages are dropped, then the last one truncated, so the
prompt fits `MAX_INPUT_TOKENS` and the model's context window.

> If your LoRA adapter was fine-tuned with a specific prompt layout, edit `INSTRUCTION` /
> `build_user_message` below to match it. Mismatched prompts reduce answer quality.
"""

CELL8_MD = """\
## Cell 8: Model inference (smoke test)

`model_service.generate_answer(question, passages)` builds the prompt, runs generation, measures
nothing it does not report, and decodes **only the newly generated tokens** (never the prompt). It raises
`InferenceError` / `InferenceTimeoutError` on failure.

This cell calls the **real loaded model** once on a short test passage so you can confirm that
inference works before starting the API. The passage below is test input only; it is not used by the
application. Edit it freely. Judge the answer yourself: this is a functional check, not an evaluation.
"""

CELL8_CODE = r'''import time

TEST_CONTEXT = (
    "The Meridian Report was published in 2021. Its main recommendation is that regional "
    "libraries extend their weekend opening hours."
)
TEST_QUESTION = "What is the main recommendation of the Meridian Report?"

_chunks = chunk_pages([PageText(1, TEST_CONTEXT)], "smoke-test", settings.chunk_size, settings.chunk_overlap)
_passages = BM25Retriever(_chunks).retrieve(TEST_QUESTION, settings.retrieval_top_k, settings.max_context_chars)

_started = time.monotonic()
try:
    _result = model_service.generate_answer(TEST_QUESTION, _passages)
except ModelServiceError as exc:
    print("INFERENCE FAILED:", exc.message)
    raise
_elapsed = time.monotonic() - _started

print("Question :", TEST_QUESTION)
print("Answer   :", _result.answer)
print(f"Model    : {_result.model_name}")
print(f"Time     : {_elapsed:.2f} s  (prompt tokens: {_result.prompt_tokens}, generated tokens: {_result.generated_tokens})")
'''

CELL9_MD = """\
## Cell 9: FastAPI application

`create_app()` builds the API. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | server status and real model readiness |
| POST | `/documents` | upload a PDF (multipart field `file`): validate, extract, chunk, store |
| GET | `/documents/{id}/text` | extracted text (bounded) for the UI's expand/copy buttons |
| POST | `/documents/{id}/questions` | retrieve chunks, build the prompt, run the SLM, return answer + evidence |
| DELETE | `/documents/{id}` | remove the document and its chunks from memory |

All errors share one JSON shape, `{"error": {"code": "...", "message": "..."}}`, and never include
stack traces or file paths (details go to the server log only). Document contents are not logged.
"""

CELL10_MD = """\
## Cell 10: CORS configuration

A browser only lets the page at `http://localhost:5173` call this API if the API names that origin.
`ALLOWED_ORIGINS` (Cell 2) is that list; `create_app()` passes it to FastAPI's `CORSMiddleware`
(allowed methods: GET, POST, DELETE, OPTIONS; no credentials).

**About the tunnel:** the browser's `Origin` is the *frontend's* origin (`http://localhost:5173`),
not the tunnel URL, so the tunnel URL does **not** go into `ALLOWED_ORIGINS`. Add an origin only if you
serve the frontend from somewhere else (for example a deployed site). A wildcard `*` is refused below.

This cell also refuses to bind the API to anything except localhost.
"""

CELL10_CODE = r'''if "*" in ALLOWED_ORIGINS:
    raise ValueError("ALLOWED_ORIGINS must list explicit origins; a wildcard '*' is not allowed.")
if API_HOST not in ("127.0.0.1", "localhost"):
    raise ValueError("API_HOST must be 127.0.0.1 or localhost. The tunnel connects locally.")

app = create_app(settings=settings, model_service=model_service)

print("FastAPI app created.")
print("Allowed browser origins:", settings.cors_origin_list)
print("Routes:", sorted({f"{sorted(r.methods - {'HEAD', 'OPTIONS'})[0]} {r.path}" for r in app.routes if hasattr(r, "methods") and r.path.startswith(("/health", "/documents"))}))
'''

CELL11_MD = """\
## Cell 11: Start the API server and the tunnel

Colab's `localhost` is **not reachable from your MacBook**, so two things happen here:

1. **uvicorn** runs the API in a background thread on `127.0.0.1:8000` (inside Colab only). It stays up
   while the Colab session is alive, and the cell returns immediately so you can keep using the notebook.
2. **cloudflared** (Cloudflare "quick tunnel") publishes that local port at a temporary
   `https://<random>.trycloudflare.com` address. No account and **no token** are needed, so there is no
   secret to protect or leak. The binary is downloaded from Cloudflare's official GitHub releases.

**Using it**
* Copy the `VITE_API_BASE_URL=...` line printed at the end into the frontend's `.env`, then restart
  `npm run dev`.
* Anyone who knows the random URL can call the API (CORS only restricts *browsers*). Treat the URL as
  private and stop the tunnel when you are done (Cell 11b).
* Cloudflare cuts requests at about 100 s, so keep `GENERATION_TIMEOUT_SECONDS` at 90 or below.
* **Restart:** re-run this cell (it stops the previous server and tunnel first). The URL changes, so
  update the frontend `.env`. If you changed Cells 2 to 10, re-run those first. If the Colab runtime
  was restarted, run everything from Cell 1.

Alternatives such as ngrok need an auth token; if you use one, keep it in Colab Secrets
(`userdata.get(...)`), never in a cell or in the frontend. They are not implemented here.
"""

CELL11_CODE = r'''import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

import uvicorn

WORK_DIR = "/content" if os.path.isdir("/content") else tempfile.gettempdir()
CLOUDFLARED_PATH = os.path.join(WORK_DIR, "cloudflared")
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
_TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

_server = None
_server_thread = None
_tunnel_process = None


def stop_server() -> None:
    global _server, _server_thread
    if _server is not None:
        _server.should_exit = True
    if _server_thread is not None:
        _server_thread.join(timeout=10)
    _server = _server_thread = None


def stop_tunnel() -> None:
    global _tunnel_process
    if _tunnel_process is not None and _tunnel_process.poll() is None:
        _tunnel_process.terminate()
        try:
            _tunnel_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _tunnel_process.kill()
    _tunnel_process = None


def start_server() -> None:
    """Run uvicorn in a background thread (a notebook cell cannot block)."""
    global _server, _server_thread
    stop_server()
    _server = uvicorn.Server(uvicorn.Config(app, host=API_HOST, port=API_PORT, log_level="info"))
    _server_thread = threading.Thread(target=_server.run, name="uvicorn", daemon=True)
    _server_thread.start()
    for _ in range(100):
        if _server.started:
            return
        if not _server_thread.is_alive():
            raise RuntimeError(f"The server failed to start. Is port {API_PORT} already in use? Re-run this cell.")
        time.sleep(0.1)
    raise RuntimeError("The server did not start within 10 seconds.")


def start_tunnel(timeout: float = 60.0) -> str:
    """Start a cloudflared quick tunnel to the local server and return its public https URL."""
    global _tunnel_process
    stop_tunnel()
    if not os.path.exists(CLOUDFLARED_PATH):
        print("Downloading cloudflared ...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED_PATH)
        os.chmod(CLOUDFLARED_PATH, 0o755)

    proc = subprocess.Popen(
        [CLOUDFLARED_PATH, "tunnel", "--no-autoupdate", "--url", f"http://{API_HOST}:{API_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _tunnel_process = proc
    lines: "queue.Queue[str]" = queue.Queue()

    def _pump() -> None:  # keep draining output so the pipe never blocks cloudflared
        for line in proc.stdout:
            lines.put(line)

    threading.Thread(target=_pump, daemon=True).start()

    recent: list[str] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            line = lines.get(timeout=1)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        recent.append(line.rstrip())
        match = _TUNNEL_URL.search(line)
        if match and match.group(0) != "https://api.trycloudflare.com":
            return match.group(0)
    stop_tunnel()
    raise RuntimeError("cloudflared did not report a tunnel URL. Last output:\n" + "\n".join(recent[-10:]))


def http_request(url: str, method: str = "GET", headers: dict | None = None, timeout: float = 15.0):
    """Tiny HTTP helper for self-checks. Returns (status, response_headers, body_text)."""
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "slm-backend-selfcheck/1.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), err.read().decode("utf-8", "replace")


# ---- start the local server and check it ------------------------------------------------
start_server()
LOCAL_URL = f"http://{API_HOST}:{API_PORT}"
status, _, body = http_request(f"{LOCAL_URL}/health")
print(f"\nLocal check  GET {LOCAL_URL}/health -> {status}\n{body}")

# ---- verify CORS from the server's point of view ------------------------------------------
for origin in [*ALLOWED_ORIGINS[:1], "https://not-allowed.example"]:
    status, hdrs, _ = http_request(
        f"{LOCAL_URL}/documents",
        method="OPTIONS",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )
    allowed = hdrs.get("access-control-allow-origin")
    print(f"CORS preflight from {origin}: HTTP {status}, allow-origin = {allowed!r}")

# ---- optional public tunnel -------------------------------------------------------------------
if START_TUNNEL:
    PUBLIC_URL = start_tunnel()
    print("\nTunnel started:", PUBLIC_URL)
    for attempt in range(15):  # a brand-new tunnel hostname can take a few seconds to resolve
        try:
            status, _, body = http_request(f"{PUBLIC_URL}/health")
            if status == 200:
                print(f"Tunnel check GET {PUBLIC_URL}/health -> {status}\n{body}")
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        print("WARNING: the tunnel did not answer /health yet. Wait a few seconds and open the URL in a browser.")
    print("\n" + "=" * 78)
    print("Put this line in frontend/.env, then restart `npm run dev`:\n")
    print(f"VITE_API_BASE_URL={PUBLIC_URL}")
    print("=" * 78)
else:
    print("\nSTART_TUNNEL is False: the API is reachable only inside Colab at", LOCAL_URL)
'''

CELL11B_MD = """\
### Cell 11b: Stop the server and the tunnel (optional)

Run this when you are finished. It stops the public URL and the API. Documents held in memory are
discarded when the session ends. To start again, re-run Cell 11.
"""

CELL11B_CODE = """stop_tunnel()
stop_server()
print("Tunnel and server stopped.")
"""


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------
def md(text: str) -> dict:
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(True),
    }


def build_cells() -> list[dict]:
    pip_args = " ".join(f'"{p}"' for p in PIP_PACKAGES)

    cell2 = join_modules(CELL2_USER_BLOCK, module_source("config").split("\n@lru_cache")[0].replace(
        "from functools import lru_cache\n", "") + CELL2_BUILD_SETTINGS)
    cell3 = join_modules(module_source("model_service"), CELL3_LOAD_CODE)
    cell5 = join_modules(module_source("chunking"), module_source("storage"))
    cell9 = join_modules(
        module_source("errors"),
        module_source("schemas"),
        module_source("main").rsplit("\napp = create_app()", 1)[0],
    )

    return [
        md(INTRO_MD),
        md(CELL1_MD), code(CELL1_CODE.replace("__PIP_ARGS__", pip_args)),
        md(CELL2_MD), code(cell2),
        md(CELL3_MD), code(cell3),
        md(CELL4_MD), code(join_modules(module_source("document_processor"))),
        md(CELL5_MD), code(cell5),
        md(CELL6_MD), code(join_modules(module_source("retrieval"))),
        md(CELL7_MD), code(join_modules(module_source("prompting"))),
        md(CELL8_MD), code(CELL8_CODE),
        md(CELL9_MD), code(cell9),
        md(CELL10_MD), code(CELL10_CODE),
        md(CELL11_MD), code(CELL11_CODE),
        md(CELL11B_MD), code(CELL11B_CODE),
    ]


def main() -> None:
    notebook = {
        "cells": build_cells(),
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "slm_document_analysis_backend.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "slm_document_analysis_backend.ipynb"
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (HERE / "requirements.txt").write_text(
        "# Packages installed by Cell 1 of the notebook (keep in sync via build_notebook.py).\n"
        "# torch is NOT listed: Colab provides a CUDA build.\n" + "\n".join(PIP_PACKAGES) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out.name} ({len(notebook['cells'])} cells) and requirements.txt")


if __name__ == "__main__":
    main()


### 🤖 Assistant

> 🔧 called: write_file, execute_command

Now I'll generate the notebook, then write a verification script that actually executes its cells, with a tiny public model and no tunnel.

> 🔧 **write_file** — """Execute the notebook's code cells locally to verify them (no Colab needed).

* Cell 1 (``%pip``) is skipped; dependencies must already be installed in the current environment.
* Cell 2 is patched to use ``MODEL_NAME_OR_PATH`` (+ optional ``ADAPTER_PATH``) from the environment,
  CPU, and ``START_TUNNEL = False``.
* Cell 11 then starts the real uvicorn server on localhost and the script exercises the API over HTTP.

Usage:  MODEL_NAME_OR_PATH=<model> [ADAPTER_PATH=<dir>] python verify_notebook.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB = json.loads((HERE / "slm_document_analysis_backend.ipynb").read_text(encoding="utf-8"))
codes = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
assert len(codes) == 12, len(codes)  # cells 1-11 plus 11b

model = os.environ["MODEL_NAME_OR_PATH"]
adapter = os.environ.get("ADAPTER_PATH", "")
codes[1] = (
    codes[1]
    .replace('MODEL_NAME_OR_PATH = ""', f"MODEL_NAME_OR_PATH = {model!r}")
    .replace('ADAPTER_PATH = ""', f"ADAPTER_PATH = {adapter!r}")
    .replace('MODEL_DEVICE = "auto"', 'MODEL_DEVICE = "cpu"')
    .replace("MAX_NEW_TOKENS = 256", "MAX_NEW_TOKENS = 8")
    .replace("START_TUNNEL = True", "START_TUNNEL = False")
    .replace("API_PORT = 8000", "API_PORT = 8771")
)

ns: dict = {"__name__": "__main__"}
for number, source in enumerate(codes):
    if number == 0:
        continue  # %pip install
    label = "11b" if number == 11 else str(number + 1)
    if number == 11:
        continue  # do not stop the server yet
    print(f"\n######## executing notebook cell {label}", flush=True)
    exec(compile(source, f"<notebook cell {label}>", "exec"), ns)

# ---- exercise the running server exactly like the frontend would ------------------------------
base = "http://127.0.0.1:8771"


def call(method: str, path: str, body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(base + path, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def multipart(name: str, data: bytes):
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={boundary}"}


import pymupdf

pdf_doc = pymupdf.open()
page = pdf_doc.new_page()
page.insert_text((72, 72), "The Meridian Report was published in 2021. Its main recommendation is longer library hours.")
pdf = pdf_doc.tobytes()

failures: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    print(("PASS " if condition else "FAIL ") + label, detail if not condition else "")
    if not condition:
        failures.append(label)


s, h = call("GET", "/health")
check("health ready", s == 200 and h["model_ready"] is True, h)
body, hdr = multipart("report.pdf", pdf)
s, up = call("POST", "/documents", body, hdr)
check("upload ok", s == 201 and up["chunk_count"] >= 1 and "Meridian" in up["text_preview"], up)
doc_id = up["document_id"]
s, t = call("GET", f"/documents/{doc_id}/text")
check("full text endpoint", s == 200 and "[Page 1]" in t["text"] and "Meridian" in t["text"], t)
s, a = call("POST", f"/documents/{doc_id}/questions", json.dumps({"question": "What does the Meridian Report recommend?"}).encode(), {"Content-Type": "application/json"})
check("question answered by model", s == 200 and a["answer_generated"] and a["answer"] and a["supporting_passages"], a)
print("   answer:", a.get("answer"), "| model:", a.get("model_name"), "| time:", a.get("inference_time_seconds"))
s, a2 = call("POST", f"/documents/{doc_id}/questions", json.dumps({"question": "zebra giraffe savannah"}).encode(), {"Content-Type": "application/json"})
check("unrelated question -> no answer, model not called", s == 200 and a2["answer_generated"] is False and a2["supporting_passages"] == [], a2)
body, hdr = multipart("bad.pdf", b"%PDF-1.4 not really a pdf")
s, e = call("POST", "/documents", body, hdr)
check("invalid pdf rejected", s == 422 and e["error"]["code"] == "malformed_pdf", e)
s, d = call("DELETE", f"/documents/{doc_id}")
check("delete ok", s == 200 and d["deleted"], d)
s, e = call("POST", f"/documents/{doc_id}/questions", json.dumps({"question": "x?"}).encode(), {"Content-Type": "application/json"})
check("question after delete -> 404", s == 404, e)

ns["stop_server"]()
print("\nNotebook verification:", "FAILED " + str(failures) if failures else "all checks passed")
sys.exit(1 if failures else 0)


> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/colab && ../slm-document-analysis-backend/.venv/bin/python build_notebook.py && ../slm-document-analysis-backend/.venv/bin/python -c \"import json;nb=json.load(open('slm_document_analysis_backend.ipynb'));print(len(nb['cells']),'cells');[print(i,c['cell_type'],len(''.join(c['source']))) for i,c in enumerate(nb['cells'])]\""

### 🤖 Assistant

> 🔧 called: execute_command

PowerShell mangles quoted Python again, so I'll keep commands simple and put any logic in script files.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/colab && ../slm-document-analysis-backend/.venv/bin/python build_notebook.py"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/colab && export MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM && /home/ardraj/ARC/Project/slm-document-analysis-backend/.venv/bin/python verify_notebook.py 2>&1 | grep -vE 'Loading weights|unauthenticated|pad_token_id|generation flags' | tail -60"

### 🤖 Assistant

> 🔧 called: apply_diff, execute_command



> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:36
-------
ns: dict = {"__name__": "__main__"}
=======
# Colab executes cells in the real __main__ module; pydantic resolves type hints through it,
# so run the cells in this script's own globals (which ARE __main__) rather than a private dict.
ns: dict = globals()
>>>>>>> REPLACE

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/colab && export MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM && /home/ardraj/ARC/Project/slm-document-analysis-backend/.venv/bin/python verify_notebook.py 2>&1 | grep -vE 'Loading weights|unauthenticated|pad_token_id|generation flags|RuntimeWarning' | tail -60"

### 🤖 Assistant

> 🔧 called: insert_content, write_file



> 🔧 **write_file** — """Temporary: verify the notebook's cloudflared tunnel code against a real quick tunnel."""
import json, os, time

src = [ "".join(c["source"]) for c in json.load(open("/home/ardraj/ARC/Project/colab/slm_document_analysis_backend.ipynb"))["cells"] if c["cell_type"] == "code"]
# Minimal app: only the tunnel/server helper cell is under test.
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

ALLOWED_ORIGINS = ["http://localhost:5173"]
API_HOST, API_PORT, START_TUNNEL = "127.0.0.1", 8772, True
helper = src[10].split("# ---- start the local server and check it")[0]
exec(helper, globals())
start_server()
url = start_tunnel()
print("TUNNEL URL:", url)
for _ in range(20):
    try:
        s, _, b = http_request(f"{url}/health")
        if s == 200:
            print("PUBLIC /health ->", s, b)
            break
    except OSError as e:
        print("not yet:", type(e).__name__)
    time.sleep(2)
else:
    print("FAILED: tunnel never answered")
stop_tunnel(); stop_server()
print("tunnel process stopped:", _tunnel_process is None)


> 🔧 **insert_content** — slm-document-analysis-backend/tests/test_api.py

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/slm-document-analysis-backend && .venv/bin/pytest -q 2>&1 | tail -5; cd ../colab && PYTHONPATH=. ../slm-document-analysis-backend/.venv/bin/python .tmp_tunnel_check.py 2>&1 | grep -vE 'RuntimeWarning|^INFO' | tail -15"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "pkill -f cloudflared; pkill -f .tmp_tunnel_check; sleep 1; pgrep -af 'cloudflared|uvicorn|tmp_tunnel' || echo no-leftovers; rm -f /home/ardraj/ARC/Project/colab/.tmp_tunnel_check.py; ls /home/ardraj/ARC/Project/colab; which chromium chromium-browser google-chrome 2>/dev/null || echo no-browser"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "ls -a /home/ardraj/ARC/Project/colab"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "rm -f /home/ardraj/ARC/Project/colab/.tmp_tunnel_check.py; ls -a /home/ardraj/ARC/Project/colab; pgrep -c cloudflared; true"

### 🤖 Assistant

> 🔧 called: update_todo_list, write_file



> 🔧 **update_todo_list** — [x] Inspect environment (no access to Mtech notebook/model config; Node via pip wheel in WSL)
[x] Scaffold React+TS+Vite frontend, install deps, typecheck/build, 13 unit tests pass
[x] Add GET /documents/{id}/text to backend (for Expand/Copy Text)
[x] Create colab/ builder script + requirements.txt
[x] Generate slm_document_analysis_backend.ipynb
[x] Verify notebook locally (cells executed with tiny model, API checks) + real cloudflared tunnel check + backend pytest (53 pass)
[-] Browser e2e of the UI against live backend (Playwright/Chromium if installable)
[ ] Fix small wording issue in notebook Cell 8 markdown, rebuild
[ ] Root README.md (setup, API, Colab, e2e test procedure)
[ ] Clean temp helper files, final summary

> 🔧 **write_file** — #!/usr/bin/env bash
# Temporary helper: try to install Playwright + Chromium (headless) inside WSL.
W="$HOME/ARC/Project/slm-document-analysis-backend/.venv/lib/python3.14/site-packages/nodejs_wheel"
export PATH="$W/bin:$PATH"
cd "$HOME/ARC/Project/frontend" || exit 1
node "$W/lib/node_modules/npm/bin/npm-cli.js" install -D playwright 2>&1 | tail -2
node node_modules/playwright/cli.js install chromium-headless-shell 2>&1 | tail -5
node node_modules/playwright/cli.js install-deps chromium-headless-shell 2>&1 | tail -3
ls ~/.cache/ms-playwright 2>&1


### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_pw.sh

### 🤖 Assistant

> 🔧 called: write_file, execute_command

Chromium downloaded, but installing its system libraries needs sudo, which isn't available. I'll check whether it launches anyway.

> 🔧 **write_file** — const { chromium } = require("playwright");
(async () => {
  try {
    const b = await chromium.launch();
    const p = await b.newPage();
    await p.setContent("<h1>ok</h1>");
    console.log("LAUNCH OK:", await p.textContent("h1"));
    await b.close();
  } catch (e) {
    console.log("LAUNCH FAILED:", String(e.message).split("\n").slice(0, 6).join("\n"));
  }
})();


> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_node.sh node .tmp_launch.cjs

### 🤖 Assistant

> 🔧 called: write_file, execute_command

Chromium is missing system libraries and I don't have sudo. I'll try fetching the needed `.deb` packages and unpacking them into a local folder, which needs no root.

> 🔧 **write_file** — #!/usr/bin/env bash
# Temporary helper: unpack missing shared libs for headless Chromium into ~/pwlibs (no root needed).
set -u
BIN=$(ls ~/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell)
LIBS=~/pwlibs
mkdir -p "$LIBS/debs" "$LIBS/root"
cd "$LIBS/debs"
for round in 1 2 3 4; do
  missing=$(LD_LIBRARY_PATH="$LIBS/root/usr/lib/x86_64-linux-gnu:$LIBS/root/lib/x86_64-linux-gnu" ldd "$BIN" 2>/dev/null | awk '/not found/ {print $1}')
  [ -z "$missing" ] && { echo "all libs resolved"; break; }
  echo "round $round missing: $(echo $missing | tr '\n' ' ')"
  for lib in $missing; do
    pkg=$(apt-file search "$lib" 2>/dev/null | head -1 | cut -d: -f1)
    if [ -z "$pkg" ]; then
      base=${lib%%.so*}; base=${base#lib}
      pkg=$(apt-cache search --names-only "^lib${base}[0-9.]*(t64)?$" 2>/dev/null | head -1 | awk '{print $1}')
    fi
    echo "  $lib -> ${pkg:-UNKNOWN}"
    [ -n "$pkg" ] && apt-get download "$pkg" 2>&1 | tail -1
  done
  for d in *.deb; do [ -f "$d" ] && dpkg -x "$d" "$LIBS/root"; done
done
LD_LIBRARY_PATH="$LIBS/root/usr/lib/x86_64-linux-gnu:$LIBS/root/lib/x86_64-linux-gnu" ldd "$BIN" | grep 'not found' || echo "no missing libs remain"


> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_libs.sh

### 🤖 Assistant

> 🔧 called: write_file, write_file



> 🔧 **write_file** — #!/usr/bin/env bash
# Temporary: start backend (tiny test model) + Vite, run the Playwright e2e, shut everything down.
P="$HOME/ARC/Project"
W="$P/slm-document-analysis-backend/.venv/lib/python3.14/site-packages/nodejs_wheel"
export PATH="$W/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/pwlibs/root/usr/lib/x86_64-linux-gnu:$HOME/pwlibs/root/lib/x86_64-linux-gnu"

"$P/slm-document-analysis-backend/.venv/bin/python" - <<'EOF'
import pymupdf
d = pymupdf.open()
p = d.new_page()
p.insert_text((72, 72), "The Meridian Report was published in 2021. Its main recommendation is that regional")
p.insert_text((72, 90), "libraries extend their weekend opening hours to improve public access.")
d.save("/tmp/e2e_report.pdf")
EOF

cd "$P/slm-document-analysis-backend"
MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM MODEL_DEVICE=cpu MAX_NEW_TOKENS=8 \
  nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 >/tmp/e2e_backend.log 2>&1 &
BACK=$!
cd "$P/frontend"
printf 'VITE_API_BASE_URL=http://127.0.0.1:8000\n' > .env
nohup node "$W/lib/node_modules/npm/bin/npm-cli.js" run dev >/tmp/e2e_vite.log 2>&1 &
sleep 14
curl -s http://127.0.0.1:8000/health | head -c 200; echo
curl -s -o /dev/null -w "vite http %{http_code}\n" http://localhost:5173/
node .tmp_e2e.cjs
CODE=$?
pkill -P $BACK 2>/dev/null; kill $BACK 2>/dev/null
pkill -f "vite" 2>/dev/null
rm -f .env
exit $CODE


> 🔧 **write_file** — // Browser end-to-end test: drives the REAL UI (Vite dev server) against the REAL backend (tiny
// test model). Run via scripts/run_e2e.sh. Not part of `npm test` (needs a backend + Chromium).
const { chromium } = require("playwright");
const fs = require("fs");

const UI = "http://localhost:5173";
const results = [];
const check = (name, ok, detail = "") => {
  results.push(ok);
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : "  -> " + detail}`);
};

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ permissions: ["clipboard-read", "clipboard-write"] });
  const page = await context.newPage();
  const consoleErrors = [];
  page.on("pageerror", (e) => consoleErrors.push(String(e)));

  await page.goto(UI);
  await page.getByRole("heading", { name: "SLM Document Analysis" }).waitFor();
  check("header, subtitle and badge", (await page.getByText("Proof of Concept").count()) === 1 &&
    (await page.getByText("Upload a document, ask questions, and explore").count()) === 1);

  // Backend health shown
  await page.getByText(/Backend connected\. Model ready/).waitFor({ timeout: 15000 });
  check("health banner shows real model name", (await page.textContent(".banner-ok")).includes("tiny-random-LlamaForCausalLM"));

  // Question box disabled before upload
  check("question disabled before upload", await page.getByRole("button", { name: "Generate Answer" }).isDisabled());

  // Invalid file type
  await page.setInputFiles('input[type="file"]', { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from("hello") });
  check("non-PDF rejected in UI", await page.getByText(/Unsupported file type/).isVisible());
  // Empty file
  await page.setInputFiles('input[type="file"]', { name: "empty.pdf", mimeType: "application/pdf", buffer: Buffer.alloc(0) });
  check("empty PDF rejected in UI", await page.getByText(/empty \(0 bytes\)/).isVisible());
  check("upload button disabled without valid file", await page.getByRole("button", { name: "Upload and Process Document" }).isDisabled());

  // Invalid-but-.pdf file is sent to the backend and rejected there
  await page.setInputFiles('input[type="file"]', { name: "fake.pdf", mimeType: "application/pdf", buffer: Buffer.from("%PDF-1.4 definitely not a real pdf") });
  await page.getByRole("button", { name: "Upload and Process Document" }).click();
  await page.getByText("Processing failed.").waitFor();
  check("backend rejects malformed PDF; message shown", (await page.textContent(".notice-error")).includes("corrupt"));
  check("question still disabled after failed upload", await page.getByRole("button", { name: "Generate Answer" }).isDisabled());

  // Valid PDF
  await page.setInputFiles('input[type="file"]', "/tmp/e2e_report.pdf");
  check("filename and size shown", (await page.textContent(".file-info")).includes("e2e_report.pdf"));
  await page.getByRole("button", { name: "Upload and Process Document" }).click();
  await page.getByText("Processing complete.").waitFor();
  check("success message", true);
  const textBox = await page.textContent(".text-box");
  check("preview shows actual extracted text", textBox.includes("Meridian Report") && textBox.includes("library"), textBox);
  const stats = await page.textContent(".stats");
  check("stats: chars/words/pages/chunks", /Characters/.test(stats) && /Words/.test(stats) && /Chunks created\s*\d/.test(stats), stats);

  // Expand + copy text
  await page.getByRole("button", { name: "Expand" }).click();
  await page.getByRole("button", { name: "Collapse" }).waitFor();
  check("expand loads full text with page markers", (await page.textContent(".text-box")).includes("[Page 1]"));
  await page.getByRole("button", { name: "Copy Text" }).click();
  await page.getByText("Copied to clipboard.").waitFor();
  check("copy text -> clipboard contains document text", (await page.evaluate(() => navigator.clipboard.readText())).includes("Meridian Report"));

  // Question 1 (answer present in the PDF)
  await page.getByPlaceholder("Ask a question about your uploaded document...").fill("What does the Meridian Report recommend?");
  await page.getByRole("button", { name: "Generate Answer" }).click();
  await page.getByRole("heading", { name: "Step 4: Generated Answer" }).waitFor();
  await page.locator(".answer").waitFor({ timeout: 60000 });
  const answer1 = await page.textContent(".answer");
  check("answer displayed (model output)", answer1.trim().length > 0, answer1);
  check("model name + inference time shown", (await page.textContent(".meta")).includes("tiny-random-LlamaForCausalLM") && /Inference time\s*[\d.]+ s/.test(await page.textContent(".meta")));
  const passage = await page.textContent(".passage");
  check("supporting evidence shown with page number", passage.includes("Page 1") && passage.includes("Meridian"), passage);
  await page.getByRole("button", { name: "Copy Answer" }).click();
  await page.getByText("Copied to clipboard.").last().waitFor();
  check("copy answer -> clipboard", (await page.evaluate(() => navigator.clipboard.readText())) === answer1);

  // Question 2 without re-uploading (example chip)
  await page.getByRole("button", { name: "What is the main topic of this document?" }).click();
  check("example chip fills the input", (await page.inputValue("#question")) === "What is the main topic of this document?");
  await page.getByRole("button", { name: "Generate Answer" }).click();
  await page.waitForFunction(() => document.querySelector(".qa-block p")?.textContent === "What is the main topic of this document?", null, { timeout: 60000 });
  check("second question answered without re-upload", true);

  // Question with no matching passage
  await page.fill("#question", "zebra giraffe savannah");
  await page.getByRole("button", { name: "Generate Answer" }).click();
  await page.getByText("No answer was generated.").waitFor({ timeout: 60000 });
  check("unanswerable question handled gracefully (no answer, no evidence)", (await page.locator(".passage").count()) === 0);

  // Reset
  await page.getByRole("button", { name: /Start New Document/ }).click();
  await page.getByText(/removed from the browser and from the backend/).waitFor();
  check("reset clears document, answer and question",
    (await page.locator(".text-box").count()) === 0 && (await page.locator(".answer").count()) === 0 &&
    (await page.inputValue("#question")) === "" && (await page.getByRole("button", { name: "Generate Answer" }).isDisabled()));
  const health = await (await fetch("http://127.0.0.1:8000/health")).json();
  check("backend confirms the document was deleted", health.documents_in_memory === 0, JSON.stringify(health));

  // Backend connection failure: block API calls and reload
  await page.route("http://127.0.0.1:8000/**", (route) => route.abort("connectionrefused"));
  await page.reload();
  await page.getByText("Backend unavailable.").waitFor();
  check("backend connection failure shows helpful message", (await page.textContent(".banner-error")).includes("Cannot reach the backend"));
  check("upload disabled while backend offline", await page.getByRole("button", { name: "Upload and Process Document" }).isDisabled());
  await page.screenshot({ path: "/tmp/e2e_offline.png", fullPage: true });

  check("no uncaught page errors", consoleErrors.length === 0, consoleErrors.join("; "));
  await browser.close();
  const failed = results.filter((r) => !r).length;
  console.log(`\n${results.length - failed}/${results.length} browser checks passed`);
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error("E2E CRASHED:", e.message); process.exit(2); });


### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_e2e.sh

### 🤖 Assistant

> 🔧 called: search_and_replace, execute_command



> 🔧 **search_and_replace** — frontend/.tmp_e2e.cjs

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash /home/ardraj/ARC/Project/.tmp_e2e.sh

### 🤖 Assistant

> 🔧 called: execute_command

All 26 browser checks pass (the earlier failure was my test's wrong expectation, not an app bug). I'll look at the screenshot to check how the UI renders.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cp /tmp/e2e_offline.png /home/ardraj/ARC/Project/.tmp_offline.png && ls -la /home/ardraj/ARC/Project/.tmp_offline.png"

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — .tmp_offline.png

### 🤖 Assistant

> 🔧 called: search_and_replace, execute_command

The UI renders correctly. The offline banner, the text-only PDF warning and the disabled controls all show as intended. The document icon appears as an empty box only because this headless Chromium has no emoji font. Next I'll fix the awkward wording in the Cell 8 markdown, rebuild the notebook, and remove the temporary files.

> 🔧 **search_and_replace** — colab/build_notebook.py

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project && rm -f .tmp_offline.png .tmp_pw.sh .tmp_libs.sh .tmp_e2e.sh .tmp_versions.sh frontend/.tmp_launch.cjs frontend/.tmp_e2e.cjs frontend/.env && bash .tmp_node.sh npm uninstall playwright 2>&1 | tail -2; rm -rf ~/pwlibs; ls -a . frontend"

### 🤖 Assistant

> 🔧 called: grep

The search didn't match, so I'll check the exact text in the file.

> 🔧 **grep** — measures

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: apply_diff, execute_command



> 🔧 **apply_diff** — colab/build_notebook.py

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/colab && ../slm-document-analysis-backend/.venv/bin/python build_notebook.py 2>/dev/null && ../slm-document-analysis-backend/.venv/bin/python -m json.tool slm_document_analysis_backend.ipynb > /dev/null && echo notebook-json-valid"

### 🤖 Assistant

> 🔧 called: write_file, insert_content

Now the root README covering setup, Colab instructions, the API reference and the end-to-end test procedure.

> 🔧 **write_file** — README.md

> 🔧 **insert_content** — slm-document-analysis-backend/README.md

### 🤖 Assistant

> 🔧 called: apply_diff, search_and_replace

That insert landed in the wrong place, between the section heading and its code block. I'll fix it and also add the new setting to `.env.example`.

> 🔧 **search_and_replace** — slm-document-analysis-backend/.env.example

> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:159
-------
## 6. Frontend example (React / Vite, origin `http://localhost:5173`)
### `GET /documents/{document_id}/text`
Extracted text with `[Page N]` markers, capped at `MAX_TEXT_RESPONSE_CHARS` (200,000). Response:
`{"document_id","text","character_count","total_character_count","truncated"}`. Errors: 400 `invalid_document_id`, 404 `document_not_found`.



```js
=======
### `GET /documents/{document_id}/text`
Extracted text with `[Page N]` markers, capped at `MAX_TEXT_RESPONSE_CHARS` (200,000). Response:
`{"document_id","text","character_count","total_character_count","truncated"}`. Errors: 400 `invalid_document_id`, 404 `document_not_found`.

## 6. Frontend example (React / Vite, origin `http://localhost:5173`)

```js
>>>>>>> REPLACE

### 🤖 Assistant

> 🔧 called: grep



> 🔧 **grep** — slm-document-analysis-backend/.env.example

### 🤖 Assistant

> 🔧 called: insert_content, read_file



> 🔧 **insert_content** — slm-document-analysis-backend/.env.example

> 🔧 **read_file** — slm-document-analysis-backend/.env.example

### 🤖 Assistant

> 🔧 called: execute_command

Final validation run of everything after the last edits, then I'll remove the last temporary helper.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd /home/ardraj/ARC/Project/slm-document-analysis-backend && .venv/bin/pytest -q 2>&1 | tail -3; cd ../colab && MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM ../slm-document-analysis-backend/.venv/bin/python verify_notebook.py 2>&1 | grep -E '^(PASS|FAIL)|Notebook verification'; cd .. && bash .tmp_node.sh npm run build 2>&1 | tail -3; bash .tmp_node.sh npm test 2>&1 | grep -E 'Tests|Test Files'; rm -rf frontend/dist .tmp_node.sh; ls -a . colab"

### 🤖 Assistant

The app is built, but I haven't opened the notebook in Google Colab. I also haven't run your model or LoRA adapter, because I can't see them. Everything was tested in Linux (WSL) with a tiny random public Llama model that produces gibberish, so answer quality is unchecked. `MODEL_NAME_OR_PATH` and `ADAPTER_PATH` are blank placeholders for you to fill in.

## Architecture
- **Frontend:** React, TypeScript and Vite on your Mac, calling the backend over HTTP.
- **Backend:** the notebook runs FastAPI and your model in Colab, and a Cloudflare quick tunnel gives it a public `https://….trycloudflare.com` URL. That tunnel needs no account and no token, so there is no secret to leak.
- **Pipeline:** the backend extracts PDF text with PyMuPDF, chunks it, ranks chunks by keyword (BM25), builds the prompt with the model's chat template, and generates the answer.

## Files created (in `/home/ardraj/ARC/Project`)
- **`frontend/`**: the UI, with the backend URL set only through `VITE_API_BASE_URL` in `.env.example`.
- **`colab/slm_document_analysis_backend.ipynb`**: your Colab backend notebook, cells 1–11 plus a stop cell (11b).
- **`colab/requirements.txt`**: the packages the notebook installs.
- **`colab/build_notebook.py`**: regenerates the notebook from the tested backend code. If you change the backend, rebuild the notebook with this rather than editing both.
- **`colab/verify_notebook.py`**: runs the notebook's cells locally with a small model.
- **`README.md`**: setup, Colab instructions, API reference and the end-to-end test procedure.
- **`slm-document-analysis-backend/`**: the earlier backend, now with a new `GET /documents/{id}/text` endpoint. The UI's Expand and Copy Text buttons need it because the upload response only has a short preview.

## What was actually tested
- **Backend:** 53 pytest tests pass, with 1 skipped that needs a real model.
- **Frontend:** 13 unit tests, the type-check and the production build pass.
- **Notebook:** I ran its cells in order and exercised the live API over HTTP (health, upload, full text, question, unrelated question, invalid PDF, delete, CORS allow and deny). All checks passed.
- **Tunnel:** I tested the notebook's tunnel code with a real Cloudflare quick tunnel on a minimal app, and its public `/health` returned 200.
- **UI in headless Chromium:** 26 of 26 checks passed against the live backend. They covered file validation, upload, preview, expand and copy, two questions, a no-match question, reset with backend deletion, and the backend-offline banner. That script was temporary and isn't included.

## Not tested
- Google Colab itself, Google Drive mounting and the Colab Secrets lookup.
- Your real model and adapter, and GPU inference.
- Chrome on macOS.
- Answer quality.
- The optional zero-shot vs. LoRA comparison, which is not implemented.

## What you need to do
1. In Colab, upload the notebook and select a GPU runtime.
2. Fill in `MODEL_NAME_OR_PATH`, and `ADAPTER_PATH` if you use one, in Cell 2. Set `MOUNT_GOOGLE_DRIVE = True` if they are on Drive. For a gated or private Hugging Face model, add `HF_TOKEN` in Colab Secrets.
3. If your adapter was trained with a specific prompt layout, match it in Cell 7 (`INSTRUCTION` and `build_user_message`).
4. Run the cells in order. Cell 11 prints a `VITE_API_BASE_URL=…` line.
5. On your Mac:
   ```bash
   cd frontend
   npm install
   cp .env.example .env     # paste the Colab URL into VITE_API_BASE_URL
   npm run dev              # open http://localhost:5173 in Chrome
   ```
6. Follow the 10-step procedure in `README.md` and treat the Colab-specific parts as unverified until you've run them.

## Limitations
- There is no OCR, so scanned PDFs are rejected with a clear message.
- Retrieval matches keywords, not meaning, so paraphrased questions can miss the right passage.
- Documents live in memory and are lost when the Colab session restarts.
- The tunnel URL changes on every restart.
- Anyone who knows the tunnel URL can call the API. There is no authentication, so keep the URL private and stop the tunnel when you're done.

I deleted all the temporary helper files. The Node install I used for testing lived inside `slm-document-analysis-backend/.venv`, which is git-ignored, so delete that folder and create a fresh venv on your Mac.

---

### 👤 User

how should i run this frontend code in local to see how it looks?

---

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — frontend/package.json

### 🤖 Assistant

The `package.json` is intact, so you can run the frontend as follows. You need Node.js 20.19 or newer, or 22.12 or newer. Check with `node --version`, and install it from <https://nodejs.org> if it's missing.

## Steps (macOS Terminal)

```bash
cd path/to/Project/frontend      # the folder containing package.json
npm install                      # one-time: downloads dependencies
cp .env.example .env             # one-time: creates your local config
npm run dev
```

Then open **<http://localhost:5173>** in Chrome.

The port is fixed at 5173 on purpose. If something else is using it, Vite exits with an error instead of switching ports, because the backend's CORS only allows that origin. Stop whatever is using the port and run `npm run dev` again.

## What you'll see without a backend

With the default `.env` (`VITE_API_BASE_URL=http://127.0.0.1:8000`) and no backend running, the page loads fully. It shows a red **"Backend unavailable"** banner with a Retry button, and Upload and Generate Answer are disabled. That is correct, and it's enough to check the layout, colors and text.

## What each part needs

To see the **full flow** (upload, extracted text, answer, evidence), you need a backend. You have two options:

1. **Colab (the intended setup):** run the notebook through Cell 11, copy the printed `VITE_API_BASE_URL=https://…trycloudflare.com` line into `frontend/.env`, then stop the dev server (`Ctrl+C`) and run `npm run dev` again. Vite only reads `.env` at startup.
2. **Locally on your Mac, without Colab:** this works only for a small model, and it's useful for trying the UI without Colab.
   ```bash
   cd slm-document-analysis-backend
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env     # set MODEL_NAME_OR_PATH (and ADAPTER_PATH if you use one)
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   Leave the frontend `.env` at its default, `http://127.0.0.1:8000`.

   For a quick UI test with a throwaway model, you can set `MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM`. It downloads in seconds but produces gibberish answers. I only used it for testing, and it has nothing to do with your project's model.

## Other commands

- `npm run build` type-checks and builds a production bundle into `dist/`.
- `npm test` runs the 13 unit tests.

If `npm install` fails, send me the error text.

---

### 👤 User

what is i am using windows terminal and not mac. can you pls give the install commands?

---

### 🤖 Assistant

Your project files are inside WSL (`\\wsl.localhost\Ubuntu\...`), so the simplest route is to run the frontend inside WSL. Windows Terminal can open an Ubuntu tab. Option B below uses native Windows instead.

## Option A: run inside WSL Ubuntu (recommended)

Node.js isn't installed in your WSL, so install it first. Open an **Ubuntu** tab in Windows Terminal and run:

```bash
# one-time: install Node via nvm (the installer's version may be newer; see github.com/nvm-sh/nvm)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install 22
node --version        # should print v22.x

# run the frontend
cd ~/ARC/Project/frontend
npm install
cp .env.example .env
npm run dev
```

Then open **<http://localhost:5173>** in your normal Windows browser. WSL2 forwards `localhost` to Windows automatically.

## Option B: native Windows (PowerShell)

```powershell
# one-time: install Node.js LTS, then CLOSE and reopen PowerShell
winget install OpenJS.NodeJS.LTS
node --version

# copy the frontend out of WSL (npm does not work well from a \\wsl.localhost path)
robocopy "\\wsl.localhost\Ubuntu\home\ardraj\ARC\Project\frontend" "$env:USERPROFILE\slm-frontend" /E /XD node_modules dist

cd $env:USERPROFILE\slm-frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open **<http://localhost:5173>**. With this option, any later edits under `\\wsl...\frontend` need to be copied again.

## What you'll see

With no backend running, the page loads and shows a red "Backend unavailable" banner. That is expected, and it's enough to check how the UI looks.

## Running the backend locally (optional, for the full flow)

In an **Ubuntu** tab:

```bash
sudo apt update && sudo apt install -y python3-venv   # needed on this WSL; venv creation failed without it earlier
cd ~/ARC/Project/slm-document-analysis-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then set MODEL_NAME_OR_PATH (and ADAPTER_PATH) in .env
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- **Existing `.venv`:** that folder already exists from my testing and works. You can skip the `python3 -m venv` line and use `source .venv/bin/activate`.
- **Windows PowerShell:** if you run the backend there, the activation command is `.venv\Scripts\Activate.ps1`.
- **Colab:** the intended setup is still the Colab notebook. Put the URL it prints in `frontend/.env`, then restart `npm run dev`.

If any step errors, paste the output and I'll fix it.

---

### 👤 User

ardraj@IBM-8W6DBY3:~/ARC/Project$ node --version
v22.22.1
ardraj@IBM-8W6DBY3:~/ARC/Project$ cp .env.example .env
cp: cannot stat '.env.example': No such file or directory
ardraj@IBM-8W6DBY3:~/ARC/Project$

---

### 🤖 Assistant

Node v22.22.1 is installed, so that part is fine. The error happened because you ran `cp` in `~/ARC/Project`. The `.env.example` file lives inside `frontend/`, so change into that folder first:

```bash
cd ~/ARC/Project/frontend
npm install
cp .env.example .env
npm run dev
```

Then open **<http://localhost:5173>** in your Windows browser.

If `npm install` complains about `node_modules`, run `rm -rf node_modules package-lock.json` first and then `npm install` again. That can happen because I installed the dependencies earlier using a different Node.

---

### 👤 User

1. delete that proof of concept things

---

### 🤖 Assistant

> 🔧 called: grep

I'm reading "that proof of concept things" as the **Proof of Concept badge** next to the title in your screenshot. I'll check where else that text appears in the frontend first.

> 🔧 **grep** — proof of concept|\.badge|className="badge"

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — frontend/src/styles.css

### 🤖 Assistant

> 🔧 called: apply_diff, apply_diff

I'll remove the badge element and its now-unused CSS rule.

> 🔧 **apply_diff** — frontend/src/styles.css

> 🔧 **apply_diff** — frontend/src/components/Header.tsx

### 🤖 Assistant

> 🔧 called: grep, execute_command

Now I'll check that nothing else references the badge. I'll also run a type-check and build, but with your Node, since I removed my temporary one.

> 🔧 **grep** — true

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/frontend && npm run build 2>&1 | tail -6"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "rm -rf ~/ARC/Project/frontend/dist; ls ~/ARC/Project/frontend"

### 🤖 Assistant

I removed the **Proof of Concept** badge from the page header. The frontend still type-checks and builds, but I haven't looked at the page in a browser. If your dev server is running, Vite reloads on its own, so the badge should disappear without a restart.

I changed `frontend/src/components/Header.tsx` and `frontend/src/styles.css`. In `Header.tsx` I deleted the badge element, and in `styles.css` I deleted the `.badge` rule that styled it.

I left the mentions of "proof of concept" in `README.md` and in the notebook's intro text. Those are descriptions, not part of the UI.

I also noticed that `frontend/` has no `.env.example` or `.gitignore` listed. Both may just have been hidden from that listing, because it only shows non-dot files. If your `cp .env.example .env` step failed earlier because of that, tell me and I'll check.

Tell me if you meant something else by "that proof of concept things". For example, I can also remove those mentions from the README and notebook, or the amber "Text-based PDFs only" warning in Step 1.

---

### 👤 User

now how do i start the backend??

---

### 🤖 Assistant

> 🔧 called: execute_command

I'll check the state of the backend folder first, so the commands match what's there.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && ls -a && ls .venv/bin/uvicorn && .venv/bin/python --version && ls .env 2>&1 | head -1"

### 🤖 Assistant

The existing `.venv` already has everything installed, so you don't need a fresh install in WSL. Open an **Ubuntu** tab in Windows Terminal. You have two options.

## Option 1: Start it with no model (fastest, UI-only check)

```bash
cd ~/ARC/Project/slm-document-analysis-backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The server starts without a model. The frontend then shows a **yellow** "Backend connected, but the model is not ready" banner. Upload, text preview and chunk count all work. Questions are disabled until a model is configured.

## Option 2: Start it with a model (full flow)

You must tell it which model to use. Nothing is assumed.

**Your own model:** create the config file and edit it.
```bash
cd ~/ARC/Project/slm-document-analysis-backend
cp .env.example .env
nano .env          # set MODEL_NAME_OR_PATH=... and ADAPTER_PATH=... ; save with Ctrl+O, Enter, Ctrl+X
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Throwaway test model (just to try the UI):** no `.env` needed.
```bash
cd ~/ARC/Project/slm-document-analysis-backend
source .venv/bin/activate
MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM MODEL_DEVICE=cpu MAX_NEW_TOKENS=8 \
  uvicorn app.main:app --host 127.0.0.1 --port 8000
```
This is a tiny random model, so it downloads in seconds and its answers are gibberish. That's expected. It only shows that the whole flow works, and it has nothing to do with your project's model.

## Checking that it works

Wait for `Application startup complete` in the terminal, then open <http://127.0.0.1:8000/health> in your browser. You should see `"status":"ok"`. With a model loaded it also shows `"model_ready":true`.

Then go to the frontend at <http://localhost:5173> and click **Retry** on the banner.

## Notes

- **Two terminals:** keep the backend and the frontend (`npm run dev`) running in separate Ubuntu tabs.
- **Model size:** a large model on CPU in WSL will be very slow or run out of memory. That is the reason the notebook uses Colab's GPU.
- **Frontend `.env`:** for this local backend, `frontend/.env` should have `VITE_API_BASE_URL=http://127.0.0.1:8000`, which is the default from `.env.example`.
- **Stopping:** press `Ctrl+C` in the backend tab.

If it errors on startup, paste the output and I'll fix it.

---

### 👤 User

can you pls help?

---

### 🤖 Assistant

Your backend is running correctly. Nothing is broken.

## What the screenshots show

- **`{"error":{"code":"http_error","message":"Not Found"}}` at `127.0.0.1:8000`:** this is expected. The API has no page at the root `/`, only the endpoints below. The terminal confirms it with `GET / ... 404`.
- **`ERROR app: Model not available at startup: MODEL_NAME_OR_PATH is not set`:** this is also expected. You started it with no model configured (Option 1), so the server starts and reports the problem instead of crashing.
- **`GET /health ... 200 OK`:** the server is up and answering.

## What to open instead

- <http://127.0.0.1:8000/health> shows the status. For now it should say `"model_ready":false`, along with the "MODEL_NAME_OR_PATH is not set" reason.
- <http://127.0.0.1:8000/docs> opens FastAPI's interactive page, where you can try every endpoint.

## Next steps

1. **Check the UI:** go to <http://localhost:5173> and click **Retry** on the red banner. It should turn into a yellow "Backend connected, but the model is not ready" banner. You can then upload a text-based PDF and see the extracted text, the counts and the chunk count. Questions stay disabled until a model is loaded.

2. **Enable questions:** stop the backend with `Ctrl+C`, then restart it with a model. To try the whole flow quickly, use the throwaway test model. Its answers are gibberish, but it shows the full path working:
   ```bash
   cd ~/ARC/Project/slm-document-analysis-backend
   source .venv/bin/activate
   MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM MODEL_DEVICE=cpu MAX_NEW_TOKENS=8 \
     uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   When the log shows `Model ready` (or `/health` says `"model_ready":true`), click **Retry** in the UI.

3. **Use your real model:** put `MODEL_NAME_OR_PATH` (and `ADAPTER_PATH`) in the backend `.env`. A large model needs a GPU, so use the Colab notebook for that.

I can also add a small page at `/` that shows the status and links to `/docs`, so it no longer shows "Not Found". Say if you want that.

---

### 👤 User

so how this works. first you need to upload and process the document. once that finishes only, you can ask questions?

---

### 🤖 Assistant

Yes. The flow is strictly in that order, and the UI enforces it.

## The flow

1. **Upload and process (Step 1).**
   - You pick a PDF and click **Upload and Process Document**.
   - The backend extracts the text page by page and splits it into chunks.
   - It keeps the chunks in memory under a new document ID and sends the ID back.
   - When the green "Processing complete" message appears, Step 2 shows the extracted text and the counts.

2. **Ask a question (Step 3).**
   - Until a document has been processed, the question box and the **Generate Answer** button are disabled.
   - When you ask, the browser sends the question together with the document ID. It does not send the PDF again.
   - The backend does three things:
     1. It finds the chunks of that document most relevant to your question.
     2. It sends only those chunks plus your question to the model.
     3. It returns the answer.
   - Step 4 shows the answer and the passages it used.

3. **Ask more questions.** You can ask as many as you like about the same document without uploading again, because the backend still holds it.

4. **Reset.** The **Start New Document / Reset** button clears the screen and deletes that document from the backend's memory.

## Why upload comes first

Reading a PDF and chunking it is a one-time job. Doing it once means each question is fast and only needs the document ID. The model also can't read a whole document at once, so it only sees the few relevant chunks.

## Two conditions for asking questions

Questions are enabled only when **both** of these are true:
- A document has been processed successfully.
- The model is loaded. With your current backend (no model set), the page shows the yellow "model is not ready" banner, so you can upload and view the text but not ask yet.

## Things that can send you back to upload

- **Backend restart or Colab disconnect:** documents are kept in memory only, so they are lost.
- **Idle for 60 minutes:** the document expires. The UI then says the document was not found and asks you to upload again.

Uploading never trains or changes the model. It only prepares the text for answering.

---

### 👤 User

I wanted you to do one thing. So in our implemented code there is a few change which we have to do. So it is like take reference of this attached notebook file. So what we are expecting is we will be uploading the document and the question along with it. It's not like two things, document upload plus question will be a single thing. And we will send that to the backend. The backend will extract the content out of it, the text, okay? But it should maintain all the structure, okay? Like heading, subheading, sections, content, paragraph, tables, all the things we should maintain the structure. And this structure, this extracted text and the user's question will be sent over to the model through a prompt. And the model will generate the answer for that, which we will send to the front end. That will be displayed over the web interface. So this is the thing. So please do the necessary changes for that.i have copied a folder names notebooks. pls go through it. that contains the .ipynb file. this folder is temporary, will be deleting it. it is not like use the same but take that as reference. below is the needful, stick strictly to it.:1. Structural Formatting in the Prompt

When your backend code extracts the tables and headers, it needs to convert them into a string format that the language model can naturally understand.
• For Tables: Convert the extracted grid data into standard Markdown tables (using | columns and - row separators) or a clean HTML/CSV string. Models like Qwen are highly trained on code and markdown, so markdown tables will keep the column-row relationships intact inside the prompt.
• For Headings: Use clean markdown tags (like # or ##) so the model recognizes the document's hierarchy.

2. Matching the Prompt Template Exactly

The structure you use to combine the text and question in your backend code must match the exact format you used during training. If your training dataset combined them using ### Document Text: and ### Question:, your live backend API must use those exact string headers. If they don't match, the model's performance will drop drastically.

3. Handling Table Coordinates (For a Wow-Factor Demo)

Examiners love it when you can prove where the answer came from. Since you aren't using RAG, a great addition to your backend script is to have the model print out its logical steps or reference text (e.g., "According to the table in Section 3..."). Since you are feeding it the unbroken 5–6 pages, the model has full visibility to do this cleanly.

Your Final M.Tech Project Architecture Diagram (Mental Check):

text
[Frontend UI] --(Uploads PDF + Type Question)--> [Backend API]
                                                       │
                                            (Extracts Text into Markdown)
                                                       │
                                            (Creates Unified Prompt String)
                                                       │
                                                       ▼
[Frontend UI] <--(Renders Text Response)------- [Qwen-0.5B + LoRA Adapter]
Use code with caution.
This workflow is clean, elegant, completely handles the token limits of your chosen 0.5B model, and keeps the scientific focus strictly on your Parameter-Efficient Fine-Tuning (PEFT) methodology. You are ready to start coding!
To get you started on the backend API, do you want to look at a small sample script using FastAPI or Flask that handles the PDF upload and formats it into the prompt string?
AI can make mistakes, so double-check responses

---

### 🤖 Assistant

> 🔧 called: glob, execute_command

I'll start by reading the `notebooks` folder you copied in.

> 🔧 **glob** — notebooks/**/*

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project && ls -la && ls -la notebooks 2>&1"

### 🤖 Assistant

> 🔧 called: office_read



> 🔧 **office_read** — notebooks/simple_pipeline.ipynb

### 🤖 Assistant

> 🔧 called: read_file, read_file

The `notebooks` folder is visible now. I'll read the `.ipynb` files directly.

> 🔧 **read_file** — notebooks/simple_pipeline.ipynb

> 🔧 **read_file** — notebooks/document_structure_extractor.ipynb

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — notebooks/build_extractor_notebook.py

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — notebooks/build_extractor_notebook.py

### 🤖 Assistant

> 🔧 called: update_todo_list, write_file



> 🔧 **update_todo_list** — [x] Read reference notebooks (extractor + training prompt: '### Context/Question/Answer', Qwen1.5-0.5B, LoRA)
[-] Backend: structure.py (elements, Markdown, answer-location lookup) + structured PDF extraction
[ ] Backend: training-format prompt, model_service (no chat template, token budget), schemas, config, POST /answer (file + question)
[ ] Backend: remove chunking/retrieval/storage + old tests; write new tests; run pytest
[ ] Verify model path with tiny model (prompt ids == training format) and, if downloadable, real Qwen1.5-0.5B
[ ] Rebuild Colab notebook + verify_notebook.py for the new pipeline
[ ] Frontend: single-submit UI (file + question), structure card, answer card; api + tests; build
[ ] Browser e2e against live backend
[ ] Update READMEs (root + backend), final summary

> 🔧 **write_file** — slm-document-analysis-backend/app/structure.py

### 🤖 Assistant

Previous conversation summary:

## Goal
The user is building an M.Tech proof of concept: **"A Study of Parameter-Efficient Fine-Tuning of Small Language Models (SLMs) for Document Analysis."**
- A React + TypeScript + Vite frontend runs on their machine.
- A FastAPI backend runs in Google Colab, exposed through a tunnel, with an optional local run in WSL.
- The model is the user's SLM, apparently **Qwen1.5-0.5B + LoRA adapter**, used for inference only.

The user's latest request is a **major architecture change**: drop the two-step "upload, then ask" flow and the retrieval pipeline. **Document and question are now sent in one request.**

## Instructions
These are the user's latest instructions, and they are strict.

**New flow:**
- `[Frontend UI] --(PDF + typed question, ONE request)--> [Backend API]`
- The backend extracts the text **preserving structure**: headings, subheadings, sections, paragraphs, lists and tables.
- The backend builds one unified prompt string from that Markdown plus the question.
- The prompt goes to `[Qwen-0.5B + LoRA]`, and the answer goes back to the frontend for display.
- **No RAG, no chunking, no retrieval.** The unbroken 5–6 pages go to the model.

**1. Structural formatting in the prompt:**
- Tables become Markdown tables (`| col |` with `---` separators).
- Headings use `#` / `##` markers.

**2. The prompt template must match the training format exactly.** The training notebook `notebooks/simple_pipeline.ipynb`, Step 3, is:
```python
def prompt(ex):
    return f"### Context:\n{ex['context'].strip()}\n\n### Question:\n{ex['question'].strip()}\n\n### Answer:\n"
```
- The model was trained to output the answer after `### Answer:\n`, with EOS appended. Unanswerable questions have the target `unanswerable`.
- Evaluation took the **first line** of the generated text as the answer, with greedy decoding (`do_sample=False`) and `max_new_tokens=32`.
- The base is `Qwen/Qwen1.5-0.5B`, a base model rather than an instruct model, so **no chat template**.
- Tokenizer: `tok.pad_token = tok.eos_token` if there is no pad token.
- LoRA: `r=8, alpha=16, dropout=0.05, target_modules=[q_proj, v_proj]`.
- The adapter was saved with `model.save_pretrained(f'{OUT}/adapter')`, e.g. `experiments/simple/adapter`.
- Training used `MAX_LEN = 384`; the new flow feeds much longer documents.

**3. "Wow factor" provenance:**
- The user's pasted text suggested the model print its logical steps or reference text.
- My chosen approach, `locate_answer` in `structure.py`, is **string-matching the answer against the extracted elements**. It reports page, section path and table row and column labels, and only when a real match exists.
- It must not be presented as model output.
- The training format makes the model emit only a short answer, so don't change the prompt to ask for reasoning, because that would break the template match.

**Reference folder:** the user copied `notebooks/` as reference only, and it will be deleted later.
- Take ideas from `document_structure_extractor.ipynb` and `build_extractor_notebook.py`, but don't reuse them verbatim or depend on them.
- Those files contain `DocumentElement` / `StructuredDocument`, a `PDFExtractor` and a `format_table_as_markdown` helper.
- The `PDFExtractor` works from font-size and bold heuristics: it takes the body font as the most common size and sets thresholds of 1.5x for title, 1.25x for H1 and 1.10x for H2.
- Its table handling uses `page.find_tables()`, blocks inside table bboxes are excluded, and items are sorted by (y, x).
- It also detects section-number regexes and bullets.

**Earlier instructions that still apply:**
- The user wants an on-screen Step 1 to Step 4 style UI.
- Never fabricate answers, metrics, model names, evidence or confidence scores.
- Never claim OCR support (text-based PDFs only).
- Describe in-memory storage honestly.
- Don't expose secrets.
- Bind to 127.0.0.1 only.
- Follow the project's `security.md`: no wildcard CORS, no hardcoded secrets, generic errors with no stack traces.
- Say what was and wasn't tested; never claim an unexecuted test passed.
- Never invent model paths. `MODEL_NAME_OR_PATH` and `ADAPTER_PATH` stay user-supplied. Now that the notebook reveals Qwen1.5-0.5B, mention it as the likely base but don't hardcode it.
- The user removed the **"Proof of Concept" badge** from the frontend header. That is done, and the build still passed.
- The user is on Windows with WSL Ubuntu. Node v22.22.1 is installed in WSL, and the dev server runs at `http://localhost:5173`.
- The user ran the existing backend locally in WSL with no model set, which gave the yellow "model not ready" state.

## Discoveries
**Environment:**
- The project root is `\\wsl.localhost\Ubuntu\home\ardraj\ARC\Project` (WSL path `~/ARC/Project`).
- File tools are blocked outside the workspace. Sibling dirs like `../Mtech` can't be read.
- `.gitignore` can't be written with `write_file` because it matches the ignore pattern; use a shell command instead.
- The user's real model/adapter location is unknown.

**Shell quirks:**
- The shell is PowerShell, and complex quoting of `wsl.exe -d Ubuntu -- bash -c "..."` breaks easily.
- Put multi-line logic in script files or use simple commands.
- I deleted my temporary Node and Playwright helpers. Node now comes from the user's own install (use `bash -lc` so nvm is loaded).

**Python environment:**
- WSL Python is 3.14.4.
- `slm-document-analysis-backend/.venv` was created with `virtualenv` (there is no `python3-venv`) and contains fastapi, uvicorn, pymupdf, pydantic-settings, pytest, httpx2, torch (CPU), transformers 5.22, peft 0.21 and accelerate.
- A throwaway model, `hf-internal-testing/tiny-random-LlamaForCausalLM`, was used for testing. It produces gibberish and is not the user's model.
- Hugging Face downloads work from WSL.
- `transformers` 5.x uses `dtype=` rather than `torch_dtype=`.
- The old `model_service` tests ran with `MODEL_DEVICE=cpu`.

**Notebook and verification:**
- The Colab notebook builder inlines `app/*.py` modules into cells and strips `from app.x import ...` lines.
- `verify_notebook.py` runs the notebook cells in the script's own `globals()` (pydantic needs `__main__`).
- A cloudflared quick tunnel (no token) was tested successfully earlier.
- Playwright and headless Chromium worked once libraries were unpacked locally. That tooling is deleted, so rebuild it if browser e2e is wanted again.

## Accomplished
**Previous deliverables (all built and tested under the OLD two-step architecture):**
- Backend project `slm-document-analysis-backend/`:
  - modules: `config`, `errors`, `schemas`, `document_processor`, `chunking`, `retrieval` (BM25), `prompting`, `model_service`, `storage`, `main`
  - 53 pytest tests passing
  - `GET /documents/{id}/text` added earlier
- `colab/`: `build_notebook.py`, generated `slm_document_analysis_backend.ipynb` (25 cells: intro plus cells 1–11 and 11b), `verify_notebook.py` and `requirements.txt`.
- `frontend/`: React/TS/Vite UI.
  - Files: `config.ts`, `types.ts`, `api.ts` (typed client with error kinds and timeouts), `utils.ts`, `clipboard.ts`, `App.tsx` and five components. The five components are Header, UploadCard, TextPreviewCard, QuestionCard and AnswerCard.
  - Also `styles.css`, 13 vitest tests, and `.env.example` with `VITE_API_BASE_URL` and `VITE_MAX_UPLOAD_MB`.
  - Browser e2e of 26 checks passed against the live old backend.
- Root `README.md` plus the backend `README.md`.

**Current refactor, just started:**
- Read all reference notebooks (done).
- **Created `slm-document-analysis-backend/app/structure.py`**, the first file of the new design. It contains:
  - `DocumentElement(type, content, page_number)`, a frozen dataclass.
  - `StructuredDocument(elements, page_count, empty_page_numbers)`, with `.markdown`, `.heading_count` and `.table_count`.
  - `table_to_markdown`, `element_to_markdown` and `elements_to_markdown`. Markers are title `#`, h1 `##`, h2 `###`, h3 `####`; list items are `- `; paragraphs are separated by blank lines.
  - `locate_answer(elements, answer, max_results=5)`, returning `AnswerLocation(page_number, element_type, section_path, snippet, table_cell: TableCellRef(value, row_label, column_label))`. It does whole-phrase normalized matching and is untested.
- A `update_todo_list` call failed because it dropped earlier completed items. Any new todo list must **preserve** the previously completed items: "Inspect environment...", "Scaffold React+TS+Vite frontend...", "Add GET /documents/{id}/text...", "Create colab/ builder script...", "Generate slm_document_analysis_backend.ipynb", "Verify notebook locally...".

**Remaining work (the plan):**
1. **Backend `document_processor.py`:** rewrite `process_pdf` to a structured extractor. Keep the validation (extension, size, `%PDF-` signature, empty, malformed or password-protected, no-meaningful-text) and the exception classes.
   - Use PyMuPDF `get_text("dict")` with font-size and bold heuristics like the reference, plus `page.find_tables()` for tables.
   - Return a `StructuredDocument` with the section-number regex and bullet detection.
   - Keep `sanitize_filename`.
2. **Backend prompt:** replace `prompting.py` with the exact training format `### Context:\n{doc}\n\n### Question:\n{q}\n\n### Answer:\n`.
   - Put the structured Markdown in the context.
   - Keep the instruction text out of the prompt, since the training format has none.
3. **`model_service.py`:**
   - No chat template; encode the prompt like training (`tok(prompt)`).
   - Decode only new tokens, take the **first line** as the answer, and stop at EOS.
   - Use greedy decoding with a small `max_new_tokens`; the default should drop from 256 to something around 32–64.
   - Budget tokens: truncate the *document* part, never the question or the `### Answer:` tail, to fit `MAX_INPUT_TOKENS` and the context window. Qwen1.5-0.5B supports 32k, but the Colab T4 memory is the limiting factor.
   - Truncate from the end, and report `document_truncated` and token counts in the response.
   - Accept an empty or `unanswerable` output as "model says unanswerable", and handle it explicitly.
   - Remove the passages parameter.
4. **`main.py`, `schemas.py` and `config.py`:**
   - One endpoint, e.g. `POST /answer` (multipart: `file` plus `question`).
   - The response carries answer, question, model_name, inference_time_seconds, extraction metadata (pages, headings, tables, characters, words), a bounded `structured_text_preview` or the Markdown (bounded), `answer_locations` from `locate_answer`, `document_truncated`, token counts and the disclaimer.
   - Keep `GET /health`, the error format and the CORS config.
   - Remove the `/documents*` endpoints, `chunking.py`, `retrieval.py`, `storage.py`, the chunk and retrieval settings (`CHUNK_*`, `RETRIEVAL_*`, `MAX_CONTEXT_CHARS`, BM25 and TTL settings), and the old tests.
   - Question validation: 1–2000 characters, not blank.
5. **Tests:** new pytest tests covering these areas.
   - Structure extraction using generated PDFs with headings and a table (e.g. `pymupdf` `insert_text` at varying font sizes, with ruled-table drawing for `find_tables`).
   - Markdown rendering.
   - `locate_answer`.
   - Prompt exact-format check.
   - API tests with a fake model (single request, validation errors, model-not-ready 503, malformed PDF, empty question).
6. **Verification:**
   - Run pytest.
   - Verify the model path with the tiny model, checking that the prompt ids equal the training format.
   - If downloadable, try real `Qwen/Qwen1.5-0.5B` on CPU for sanity (no adapter exists locally).
7. **Colab notebook:** update `build_notebook.py` for the new module set and cell descriptions, and drop the chunking, retrieval and storage cells.
   - Note the training format and Qwen defaults.
   - Update `verify_notebook.py` so its checks use the single endpoint.
   - Rebuild the notebook, re-run the verification, and re-check the cloudflared helper.
8. **Frontend:** convert to a single-submit UI.
   - A single card with file picker or drag-and-drop plus the question box and one "Get Answer" button, with the loading state.
   - Show the extracted-structure view (Markdown preview with counts), the answer card (answer, question, model, inference time, copy button), "Where the answer text appears in the document" from `answer_locations`, a truncation warning, and a reset that only clears the browser (nothing is stored server-side).
   - Update `types.ts`, `api.ts` (`askAboutDocument(file, question)`) and the tests.
   - Remove Expand and Copy Text if unnecessary, or keep a copy of the structured text from the response.
   - Keep the header without the badge.
   - Keep the health banner, the 'text-based PDFs only' warning and the error handling.
   - Run `npm run build` and `npm test`.
9. **Browser e2e (optional):** rebuild Playwright and the libs if needed, then update the READMEs (root and backend) with the new endpoint, request/response format, Colab steps and an honest tested/untested section. Update `.env.example`.
10. **Cleanup:**
    - Remove temp files.
    - Remove `frontend/dist`.
    - Don't delete the user's `notebooks/` folder (the user will).
    - Write the final summary stating what was tested versus not.

**Still untested overall:** Colab itself, the user's real adapter, GPU, Drive, Chrome on Windows (the user ran the old UI manually and it rendered fine), and answer quality.

## Relevant files / directories
**Root:** `~/ARC/Project/` (`\\wsl.localhost\Ubuntu\home\ardraj\ARC\Project`)
- `README.md` (root; documents the OLD two-step flow, to be rewritten)
- `notebooks/` (user's temporary reference folder; contains `document_structure_extractor.ipynb`, `build_extractor_notebook.py`, `simple_pipeline.ipynb`, `build_simple_notebook.py`, and `*:Zone.Identifier` files; don't modify)

**Backend:** `slm-document-analysis-backend/`
- `app/structure.py` (NEW, just created; needs tests)
- `app/document_processor.py` (to rewrite; currently flat per-page text, `PageText` and `ExtractedDocument`)
- `app/prompting.py` (to rewrite)
- `app/model_service.py` (to adapt)
- `app/main.py`, `app/schemas.py`, `app/config.py`, `app/errors.py` (to adapt; `errors.py` is fine)
- `app/chunking.py`, `app/retrieval.py`, `app/storage.py` (to delete)
- `tests/conftest.py`, `tests/test_api.py`, `tests/test_chunking.py`, `tests/test_document_processor.py`, `tests/test_model_integration.py` (to rewrite or delete)
- `.env.example`, `requirements.txt`, `pytest.ini`, `scripts/smoke_test.sh` (update the script to the new endpoint)
- `README.md`, `.venv/`, `.gitignore`

**Colab:** `colab/`
- `build_notebook.py`, `verify_notebook.py`, `slm_document_analysis_backend.ipynb` (generated), `requirements.txt`

**Frontend:** `frontend/` (`package.json`, `vite.config.ts`, `tsconfig.json`, `index.html`, `.env.example`, `.gitignore`, `node_modules`)
- `src/`: `main.tsx`, `App.tsx`, `api.ts`, `api.test.ts`, `types.ts`, `config.ts`, `utils.ts`, `utils.test.ts`, `clipboard.ts`, `styles.css`, `vite-env.d.ts`
- `src/components/`: `Header.tsx`, `UploadCard.tsx`, `TextPreviewCard.tsx`, `QuestionCard.tsx`, `AnswerCard.tsx`



---

### 👤 User

continue

---

### 🤖 Assistant

> 🔧 called: list_files, read_file, read_file

I'll start by reviewing the current backend files so the rewrite fits what's there.

> 🔧 **read_file** — slm-document-analysis-backend/app/document_processor.py

> 🔧 **read_file** — slm-document-analysis-backend/app/structure.py

> 🔧 **list_files** — slm-document-analysis-backend

### 🤖 Assistant

> 🔧 called: read_file, read_file, read_file, read_file, read_file, read_file



> 🔧 **read_file** — slm-document-analysis-backend/app/model_service.py

> 🔧 **read_file** — slm-document-analysis-backend/app/main.py

> 🔧 **read_file** — slm-document-analysis-backend/app/config.py

> 🔧 **read_file** — slm-document-analysis-backend/app/schemas.py

> 🔧 **read_file** — slm-document-analysis-backend/app/prompting.py

> 🔧 **read_file** — slm-document-analysis-backend/app/errors.py

### 🤖 Assistant

> 🔧 called: apply_diff

Now I'll rewrite the extractor. First, a small tweak in `structure.py`: bullet glyphs get normalised to `- `, numbered items stay as they are.

> 🔧 **apply_diff** — slm-document-analysis-backend/app/structure.py

### 🤖 Assistant

> 🔧 called: apply_diff



> 🔧 **apply_diff** — slm-document-analysis-backend/app/structure.py

### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — slm-document-analysis-backend/app/document_processor.py

### 🤖 Assistant

> 🔧 called: write_file, read_file

Next the prompt module. It uses the exact training template and nothing else.

> 🔧 **write_file** — """Prompt construction. The layout MUST match the layout used for fine-tuning.

The training notebook (``simple_pipeline.ipynb``, Step 3) formatted every example as::

    ### Context:
    {context}

    ### Question:
    {question}

    ### Answer:
    {answer}<EOS>

At inference the same text is built up to and including ``### Answer:\\n`` and the model
continues from there. No instruction sentence and no chat template is added, because the
adapter never saw them during training; adding them would move the prompt away from the
training distribution.

Here ``{context}`` is the whole document as structure-preserving Markdown (headings as
``#``/``##``, tables as ``| a | b |`` rows, list items as ``- item``).
"""

from __future__ import annotations

CONTEXT_HEADER = "### Context:\n"
QUESTION_HEADER = "\n\n### Question:\n"
ANSWER_HEADER = "\n\n### Answer:\n"


def build_prompt(context: str, question: str) -> str:
    """Return the training-format prompt (ending right after ``### Answer:\\n``)."""
    return f"{CONTEXT_HEADER}{context.strip()}{QUESTION_HEADER}{question.strip()}{ANSWER_HEADER}"


def prompt_parts(context: str, question: str) -> tuple[str, str]:
    """Return ``(prefix, suffix)`` where ``prefix + context + suffix == build_prompt(...)``.

    Used by the model service to truncate only the document part when the prompt would
    exceed the token budget, while keeping the question and ``### Answer:`` intact.
    """
    suffix = f"{QUESTION_HEADER}{question.strip()}{ANSWER_HEADER}"
    return CONTEXT_HEADER, suffix


def first_line(generated: str) -> str:
    """The evaluation in the notebook takes the first line of the generated text as the answer."""
    stripped = generated.strip()
    return stripped.splitlines()[0].strip() if stripped else ""


def is_unanswerable(answer: str) -> bool:
    """The training data uses the literal target ``unanswerable`` for unanswerable questions."""
    return answer.strip().strip(".").lower() == "unanswerable"


> 🔧 **read_file** — notebooks/simple_pipeline.ipynb

### 🤖 Assistant

> 🔧 called: grep

The `write_file` call lost its path, so I'll redo it. First I'll check how the notebook tokenised the prompt, since inference has to match it.

> 🔧 **grep** — tok\(|tokenizer\(|eos|add_special|max_new_tokens|generate\(|split\('\\n'\)|splitlines|padding_side|truncat

### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — slm-document-analysis-backend/app/prompting.py

### 🤖 Assistant

> 🔧 called: write_file

Now the model service. It encodes the prompt as one string, the way training did. If the prompt is too long, it shortens only the document part.

> 🔧 **write_file** — """Model service: loads the configured SLM (+ optional LoRA adapter) once and answers questions.

Kept independent of PDF extraction. The rest of the app only depends on the small
``AnswerGenerator`` protocol below, so a remote inference implementation (e.g. an HTTP
client for a model hosted elsewhere) could replace ``ModelService`` without touching other
modules. No remote implementation exists.

Inference mirrors the training/evaluation notebook:

* the prompt is the plain-text training template (no chat template),
* decoding is greedy (``do_sample=False``),
* only newly generated tokens are decoded,
* the FIRST LINE of the generated text is the answer.

Heavy libraries (torch / transformers / peft) are imported lazily inside ``load`` so the
unit tests and the API can start without them being initialised.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from app.config import Settings
from app.prompting import build_prompt, first_line, is_unanswerable, prompt_parts

logger = logging.getLogger(__name__)


# --- Errors -------------------------------------------------------------------
class ModelServiceError(Exception):
    code = "model_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ModelNotConfiguredError(ModelServiceError):
    code = "model_not_configured"


class ModelNotReadyError(ModelServiceError):
    code = "model_not_ready"


class ModelLoadError(ModelServiceError):
    code = "model_load_failed"


class InferenceError(ModelServiceError):
    code = "inference_failed"


class InferenceTimeoutError(ModelServiceError):
    code = "inference_timeout"


# --- Public result / interface --------------------------------------------------
@dataclass(frozen=True)
class GenerationResult:
    answer: str                 # first line of the generated text (may be "unanswerable")
    unanswerable: bool          # the model produced the training target "unanswerable"
    model_name: str
    prompt_tokens: int
    generated_tokens: int
    document_truncated: bool    # the document text had to be shortened to fit the model
    document_chars_total: int
    document_chars_used: int
    device: str
    dtype: str


class AnswerGenerator(Protocol):
    """Interface the API depends on."""

    @property
    def is_ready(self) -> bool: ...

    @property
    def model_name(self) -> str: ...

    @property
    def status(self) -> dict[str, Any]: ...

    def generate_answer(self, document_markdown: str, question: str) -> GenerationResult: ...


# --- Implementation -----------------------------------------------------------
class ModelService:
    """Local Transformers (+ PEFT) inference. The model is loaded once and reused."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = "unloaded"
        self._dtype = "unloaded"
        self._adapter_loaded = False
        self._load_error: Optional[str] = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()  # one generation at a time

    # -- state -------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def model_name(self) -> str:
        base = self._settings.model_name_or_path or "unconfigured"
        # Local paths are reduced to their last component to avoid leaking filesystem layout.
        if os.path.isabs(base) or base.startswith(("~", ".")):
            base = os.path.basename(os.path.normpath(os.path.expanduser(base)))
        if self._adapter_loaded and self._settings.adapter_path:
            adapter = os.path.basename(os.path.normpath(self._settings.adapter_path))
            return f"{base} + LoRA adapter '{adapter}'"
        return base

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.is_ready,
            "model_name": self.model_name if self._settings.model_name_or_path else None,
            "adapter_loaded": self._adapter_loaded,
            "device": self._device,
            "load_error": self._load_error,
        }

    # -- loading -----------------------------------------------------------------
    def load(self) -> None:
        """Load tokenizer, base model and (optionally) the LoRA adapter.

        Raises ``ModelNotConfiguredError`` or ``ModelLoadError``. Safe to call again;
        it does nothing if the model is already loaded.
        """
        with self._load_lock:
            if self.is_ready:
                return
            s = self._settings
            if not s.model_name_or_path:
                msg = (
                    "MODEL_NAME_OR_PATH is not set. Set it (and ADAPTER_PATH if you use a "
                    "LoRA adapter) in the .env file."
                )
                self._load_error = msg
                raise ModelNotConfiguredError(msg)
            if s.adapter_path and not os.path.isdir(os.path.expanduser(s.adapter_path)):
                msg = "ADAPTER_PATH does not point to an existing directory."
                self._load_error = msg
                raise ModelLoadError(msg)

            try:
                self._load_unchecked()
                self._load_error = None
            except ModelServiceError as exc:
                self._load_error = exc.message
                raise
            except Exception as exc:
                logger.exception("Model loading failed")
                self._model = None
                self._tokenizer = None
                self._load_error = f"Model loading failed ({type(exc).__name__}). See server logs."
                raise ModelLoadError(self._load_error) from exc

    def _load_unchecked(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self._settings
        device = self._resolve_device(torch)
        dtype_name, dtype = self._resolve_dtype(torch, device)
        logger.info("Loading model on device=%s dtype=%s", device, dtype_name)

        tokenizer = AutoTokenizer.from_pretrained(
            s.tokenizer_name_or_path or s.model_name_or_path,
            trust_remote_code=s.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            s.model_name_or_path,
            dtype=dtype,
            trust_remote_code=s.trust_remote_code,
        )

        adapter_loaded = False
        if s.adapter_path:
            from peft import PeftModel

            logger.info("Loading LoRA adapter")
            model = PeftModel.from_pretrained(model, os.path.expanduser(s.adapter_path))
            if s.merge_adapter:
                model = model.merge_and_unload()
            adapter_loaded = True

        model.to(device)
        model.eval()

        # Same as the training notebook.
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._tokenizer = tokenizer
        self._model = model
        self._device = device
        self._dtype = dtype_name
        self._adapter_loaded = adapter_loaded
        logger.info("Model ready")

    def _resolve_device(self, torch: Any) -> str:
        requested = self._settings.model_device
        if requested == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if requested == "cuda" and not torch.cuda.is_available():
            raise ModelLoadError("MODEL_DEVICE=cuda but CUDA is not available on this machine.")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ModelLoadError("MODEL_DEVICE=mps but Apple MPS is not available on this machine.")
        return requested

    def _resolve_dtype(self, torch: Any, device: str) -> tuple[str, Any]:
        requested = self._settings.model_dtype
        if requested == "auto":
            requested = "float32" if device == "cpu" else "float16"
        return requested, getattr(torch, requested)

    # -- inference -----------------------------------------------------------------
    def generate_answer(self, document_markdown: str, question: str) -> GenerationResult:
        """Build the training-format prompt, generate greedily and decode only new tokens."""
        if not self.is_ready:
            raise ModelNotReadyError(
                self._load_error or "The model is not loaded yet. Check /health for details."
            )
        import torch

        s = self._settings
        try:
            input_ids, used_chars = self._encode_within_budget(document_markdown, question)
        except ModelServiceError:
            raise
        except Exception as exc:
            logger.exception("Prompt construction failed")
            raise InferenceError("Failed to build the model prompt.") from exc
        prompt_tokens = int(input_ids.shape[-1])

        started = time.monotonic()
        try:
            with self._generate_lock, torch.inference_mode():
                output = self._model.generate(
                    input_ids=input_ids.to(self._device),
                    attention_mask=torch.ones_like(input_ids).to(self._device),
                    max_new_tokens=s.max_new_tokens,
                    max_time=s.generation_timeout_seconds,
                    do_sample=False,  # greedy, as in the notebook's evaluation
                    pad_token_id=self._tokenizer.pad_token_id,
                )
        except Exception as exc:
            logger.exception("Generation failed")
            raise InferenceError(f"Generation failed ({type(exc).__name__}).") from exc
        elapsed = time.monotonic() - started

        new_tokens = output[0][prompt_tokens:]
        generated_tokens = int(new_tokens.shape[-1])
        # max_time stops generation early; an unfinished, timed-out run is a timeout.
        hit_eos = generated_tokens > 0 and int(new_tokens[-1]) == self._tokenizer.eos_token_id
        if (
            elapsed >= s.generation_timeout_seconds
            and not hit_eos
            and generated_tokens < s.max_new_tokens
        ):
            raise InferenceTimeoutError(
                f"Answer generation exceeded {s.generation_timeout_seconds:.0f} seconds."
            )

        decoded = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        answer = first_line(decoded)
        if not answer:
            raise InferenceError("The model produced an empty answer.")

        return GenerationResult(
            answer=answer,
            unanswerable=is_unanswerable(answer),
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            document_truncated=used_chars < len(document_markdown.strip()),
            document_chars_total=len(document_markdown.strip()),
            document_chars_used=used_chars,
            device=self._device,
            dtype=self._dtype,
        )

    def _token_budget(self) -> int:
        """Max prompt tokens: MAX_INPUT_TOKENS, capped by the model's context window."""
        s = self._settings
        limit = getattr(self._tokenizer, "model_max_length", None)
        if not isinstance(limit, int) or limit > 10_000_000:
            limit = getattr(getattr(self._model, "config", None), "max_position_embeddings", None)
        budget = s.max_input_tokens
        if isinstance(limit, int) and 0 < limit <= 10_000_000:
            budget = min(budget, limit - s.max_new_tokens)
        return budget

    def _encode(self, text: str) -> Any:
        # Same call as the training notebook: tok(prompt)["input_ids"] (default special tokens).
        return self._tokenizer(text, return_tensors="pt")["input_ids"]

    def _encode_within_budget(self, document: str, question: str) -> tuple[Any, int]:
        """Encode the full prompt; if too long, cut the END of the document (never the
        question or ``### Answer:``). Returns ``(input_ids, document_chars_used)``."""
        document = document.strip()
        budget = self._token_budget()
        ids = self._encode(build_prompt(document, question))
        if ids.shape[-1] <= budget:
            return ids, len(document)

        prefix, suffix = prompt_parts(question)

        def fits(n: int) -> tuple[bool, Any]:
            encoded = self._encode(prefix + document[:n].rstrip() + suffix)
            return encoded.shape[-1] <= budget, encoded

        if not fits(0)[0]:
            raise InferenceError(
                "The question is too long to fit the model's input limit. Shorten it."
            )
        low, high = 0, len(document)  # largest n with fits(n); fits(0) is True, fits(high) False
        while high - low > 1:
            mid = (low + high) // 2
            if fits(mid)[0]:
                low = mid
            else:
                high = mid
        # Avoid cutting inside a word: back up to the previous whitespace when possible.
        cut = low
        boundary = max(document.rfind(" ", 0, cut), document.rfind("\n", 0, cut))
        if boundary > cut * 0.9:
            cut = boundary
        ok, encoded = fits(cut)
        if not ok:  # extremely unlikely; fall back to the exact fitting length
            cut = low
            _, encoded = fits(cut)
        logger.info("Document truncated to %d of %d characters", cut, len(document))
        return encoded, len(document[:cut].rstrip())


### 🤖 Assistant

> 🔧 called: apply_diff

Next is the config. I'm removing the chunking, retrieval and storage settings, and dropping sampling because decoding is now always greedy.

> 🔧 **apply_diff** — slm-document-analysis-backend/app/config.py

### 🤖 Assistant

> 🔧 called: write_file, write_file

Now the schemas and `main.py`. The endpoint is a single `POST /answer` that takes multipart form data.

> 🔧 **write_file** — """FastAPI application: upload a PDF and a question in ONE request, get an answer.

Flow of ``POST /answer``::

    PDF + question --> structure-preserving extraction (Markdown) --> training-format prompt
                   --> SLM (+ LoRA adapter) --> answer

There is no retrieval, no chunking and no server-side storage: nothing is kept after the
response is sent.

Run with:  ``uvicorn app.main:app --host 127.0.0.1 --port 8000``
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, get_settings
from app.document_processor import (
    DocumentProcessingError,
    EmptyFileError,
    FileTooLargeError,
    MalformedPDFError,
    NoExtractableTextError,
    TooManyPagesError,
    UnsupportedFileTypeError,
    process_pdf,
    sanitize_filename,
)
from app.errors import ApiError
from app.model_service import (
    AnswerGenerator,
    InferenceTimeoutError,
    ModelNotConfiguredError,
    ModelNotReadyError,
    ModelService,
    ModelServiceError,
)
from app.schemas import (
    AnswerLocation,
    AnswerResponse,
    DocumentInfo,
    ErrorResponse,
    HealthResponse,
    TableCell,
)
from app.structure import locate_answer

logger = logging.getLogger("app")

_UPLOAD_STATUS = {
    UnsupportedFileTypeError: 415,
    EmptyFileError: 400,
    FileTooLargeError: 413,
    TooManyPagesError: 413,
    MalformedPDFError: 422,
    NoExtractableTextError: 422,
}

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    415: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    504: {"model": ErrorResponse},
}


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, content={"error": {"code": code, "message": message}}
    )


def create_app(
    settings: Optional[Settings] = None,
    model_service: Optional[AnswerGenerator] = None,
) -> FastAPI:
    """Application factory. ``settings`` and ``model_service`` can be injected (used by tests)."""
    settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    service: AnswerGenerator = model_service or ModelService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # Only the built-in ModelService is loaded here; injected services manage themselves.
        if isinstance(service, ModelService) and settings.load_model_on_startup:
            try:
                await run_in_threadpool(service.load)
            except ModelServiceError as exc:
                # The API still starts so /health can report the problem clearly.
                logger.error("Model not available at startup: %s", exc.message)
        yield

    app = FastAPI(
        title="SLM Document Analysis API",
        description="Send a PDF and a question in one request; a small language model answers "
        "from the document's structure-preserving text. Nothing is stored on the server.",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        allow_credentials=False,
    )

    # --- error handlers ------------------------------------------------------
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Report only field location and a short message; never echo the submitted input.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}"
            for e in exc.errors()
        )
        return _error_response(422, "validation_error", f"Invalid request: {problems}")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled error", exc_info=exc)
        return _error_response(500, "internal_error", "An unexpected error occurred.")

    # --- endpoints -------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        status = service.status
        return HealthResponse(
            status="ok",
            model_ready=bool(status.get("model_ready")),
            model_name=status.get("model_name"),
            adapter_loaded=bool(status.get("adapter_loaded")),
            device=status.get("device"),
            model_error=status.get("load_error"),
        )

    @app.post(
        "/answer",
        response_model=AnswerResponse,
        responses=_ERROR_RESPONSES,
        tags=["question answering"],
    )
    async def answer_question(
        file: UploadFile = File(..., description="A text-based PDF."),
        question: str = Form(..., description="The question about the PDF."),
    ) -> AnswerResponse:
        question = question.strip()
        if not question:
            raise ApiError(422, "empty_question", "The question must not be empty.")
        if len(question) > settings.max_question_chars:
            raise ApiError(
                422,
                "question_too_long",
                f"The question must be at most {settings.max_question_chars} characters.",
            )

        # Fail fast, before spending time on extraction, if the model cannot answer.
        if not service.is_ready:
            status = service.status
            raise ApiError(
                503,
                "model_not_ready",
                status.get("load_error")
                or "The language model is not ready. Check GET /health for details.",
            )

        # Read at most max_bytes + 1 so oversized uploads are detected without buffering them all.
        data = await file.read(settings.max_upload_bytes + 1)
        filename = sanitize_filename(file.filename)
        try:
            structured = await run_in_threadpool(
                process_pdf,
                file.filename,
                data,
                max_bytes=settings.max_upload_bytes,
                min_text_chars=settings.min_text_chars,
                max_pages=settings.max_pages,
            )
        except DocumentProcessingError as exc:
            status_code = next(
                (s for cls, s in _UPLOAD_STATUS.items() if isinstance(exc, cls)), 422
            )
            logger.info("Upload rejected: %s", exc.code)
            raise ApiError(status_code, exc.code, exc.message) from exc
        except Exception as exc:
            logger.exception("Unexpected failure while processing PDF")
            raise ApiError(500, "processing_failed", "The document could not be processed.") from exc

        markdown = structured.markdown
        logger.info(
            "Document extracted pages=%d headings=%d tables=%d chars=%d",
            structured.page_count, structured.heading_count, structured.table_count, len(markdown),
        )

        try:
            started = time.monotonic()
            result = await run_in_threadpool(service.generate_answer, markdown, question)
            elapsed = time.monotonic() - started
        except InferenceTimeoutError as exc:
            raise ApiError(504, exc.code, exc.message) from exc
        except (ModelNotReadyError, ModelNotConfiguredError) as exc:
            raise ApiError(503, exc.code, exc.message) from exc
        except ModelServiceError as exc:
            raise ApiError(500, exc.code, exc.message) from exc

        # Plain string matching of the answer against the extracted elements (not model output).
        locations = []
        if not result.unanswerable:
            locations = [
                AnswerLocation(
                    page_number=loc.page_number,
                    element_type=loc.element_type,
                    section_path=list(loc.section_path),
                    snippet=loc.snippet,
                    table_cell=(
                        TableCell(
                            value=loc.table_cell.value,
                            row_label=loc.table_cell.row_label,
                            column_label=loc.table_cell.column_label,
                        )
                        if loc.table_cell
                        else None
                    ),
                )
                for loc in locate_answer(structured.elements, result.answer)
            ]

        limit = settings.max_markdown_response_chars
        message: Optional[str] = None
        if result.unanswerable:
            message = (
                "The model answered 'unanswerable': it judged that the document does not "
                "contain the answer."
            )
        elif result.document_truncated:
            message = (
                "The document was longer than the model's input limit, so only its beginning "
                "was used. The answer may be missing information from the cut-off part."
            )

        return AnswerResponse(
            question=question,
            answer=result.answer,
            unanswerable=result.unanswerable,
            message=message,
            model_name=result.model_name,
            inference_time_seconds=round(elapsed, 3),
            prompt_tokens=result.prompt_tokens,
            generated_tokens=result.generated_tokens,
            document=DocumentInfo(
                filename=filename,
                page_count=structured.page_count,
                heading_count=structured.heading_count,
                table_count=structured.table_count,
                character_count=len(markdown),
                word_count=len(markdown.split()),
                empty_page_numbers=structured.empty_page_numbers,
            ),
            document_truncated=result.document_truncated,
            document_chars_used=result.document_chars_used,
            document_chars_total=result.document_chars_total,
            structured_text=markdown[:limit],
            structured_text_truncated=len(markdown) > limit,
            answer_locations=locations,
        )

    return app


app = create_app()


> 🔧 **write_file** — """Pydantic response models (the public API contract)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(examples=["ok"])
    model_ready: bool
    model_name: Optional[str] = None
    adapter_loaded: bool = False
    device: Optional[str] = None
    model_error: Optional[str] = None


class DocumentInfo(BaseModel):
    """Facts measured while extracting the uploaded PDF."""

    filename: str
    page_count: int
    heading_count: int
    table_count: int
    character_count: int = Field(description="Characters in the structured (Markdown) text.")
    word_count: int
    empty_page_numbers: list[int] = Field(
        description="1-based pages from which no text could be extracted."
    )


class TableCell(BaseModel):
    value: str
    row_label: Optional[str] = Field(None, description="First-column value of the matching row.")
    column_label: Optional[str] = Field(None, description="Header text of the matching column.")


class AnswerLocation(BaseModel):
    """Where the answer text literally occurs in the extracted document.

    Found by plain string matching of the model's answer against the extracted elements.
    It is NOT produced by the model and is not a citation of the model's reasoning.
    """

    page_number: int
    element_type: str = Field(
        description="paragraph | list_item | table | title | heading_1 | heading_2 | heading_3"
    )
    section_path: list[str] = Field(description="Headings above the match, outermost first.")
    snippet: str
    table_cell: Optional[TableCell] = None


class AnswerResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    question: str
    answer: str = Field(description="First line of the text generated by the model.")
    unanswerable: bool = Field(
        description="True when the model produced its training label 'unanswerable', i.e. it "
        "judged that the document does not contain the answer."
    )
    message: Optional[str] = None
    model_name: str
    inference_time_seconds: float
    prompt_tokens: int
    generated_tokens: int

    document: DocumentInfo
    document_truncated: bool = Field(
        description="True when the document did not fit the model's input limit and its "
        "END was cut. The model then answered from the beginning only."
    )
    document_chars_used: int
    document_chars_total: int

    structured_text: str = Field(description="The Markdown given to the model (bounded).")
    structured_text_truncated: bool

    answer_locations: list[AnswerLocation]
    disclaimer: str = (
        "Answers are generated by a small language model and may be incorrect. "
        "Check them against the document."
    )


### 🤖 Assistant

> 🔧 called: write_file

`main.py` didn't get written because the path was missing again. I'll write it now.

> 🔧 **write_file** — slm-document-analysis-backend/app/main.py

### 🤖 Assistant

> 🔧 called: execute_command

Now I'm removing the obsolete modules and tests, and checking the old test fixtures.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && rm -f app/chunking.py app/retrieval.py app/storage.py tests/test_chunking.py tests/test_document_processor.py tests/test_api.py && ls app tests && cat tests/conftest.py && sed -n 1,200p tests/test_model_integration.py"

### 🤖 Assistant

> 🔧 called: write_file, execute_command

Now the test fixtures. They include a PDF generator with a title, headings, a bulleted list and a ruled table, so the extractor gets exercised properly.

> 🔧 **write_file** — """Shared test fixtures: generated PDFs and a fake model service (no real model needed)."""

from __future__ import annotations

from typing import Any

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_service import GenerationResult, ModelNotReadyError


def make_pdf(pages: list[str]) -> bytes:
    """Plain text-based PDF where each string becomes one page (single font size)."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 550, 780), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def make_structured_pdf() -> bytes:
    """A 2-page PDF with a title, headings, a paragraph, bullets and a ruled table.

    Page 1: title (24pt), "1. Introduction" (16pt), body paragraph (11pt), "1.1 Scope"
            (13pt), two bullet items.
    Page 2: "2. Results" (16pt), a ruled 3x3 table, a closing sentence.
    """
    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text((50, 80), "Annual Report 2024", fontsize=24, fontname="hebo")
    page.insert_text((50, 130), "1. Introduction", fontsize=16, fontname="hebo")
    body = [
        "This report describes the performance of the company during the year.",
        "Revenue increased because of strong demand in the northern region.",
    ]
    for i, line in enumerate(body):
        page.insert_text((50, 155 + i * 14), line, fontsize=11)
    page.insert_text((50, 215), "1.1 Scope", fontsize=13, fontname="hebo")
    page.insert_text((50, 240), "\u2022 Financial results", fontsize=11)
    page.insert_text((50, 254), "\u2022 Customer satisfaction", fontsize=11)

    page2 = doc.new_page()
    page2.insert_text((50, 80), "2. Results", fontsize=16, fontname="hebo")
    left, top, row_h = 50, 110, 24
    col_w = [140, 100, 100]
    rows = [["Region", "Revenue", "Profit"], ["North", "120", "30"], ["South", "90", "15"]]
    right = left + sum(col_w)
    for r in range(len(rows) + 1):  # horizontal rules
        y = top + r * row_h
        page2.draw_line((left, y), (right, y))
    x = left
    for w in [0] + col_w:  # vertical rules
        x += w
        page2.draw_line((x, top), (x, top + len(rows) * row_h))
    for r, row in enumerate(rows):
        x = left
        for c, cell in enumerate(row):
            page2.insert_text((x + 6, top + r * row_h + 16), cell, fontsize=11)
            x += col_w[c]
    page2.insert_text((50, top + len(rows) * row_h + 40), "The north region was the most profitable.", fontsize=11)

    data = doc.tobytes()
    doc.close()
    return data


class FakeModelService:
    """Stands in for ModelService. Records the document and question it was given."""

    def __init__(self, ready: bool = True, answer: str = "fake answer", unanswerable: bool = False) -> None:
        self.ready = ready
        self.answer = answer
        self.unanswerable = unanswerable
        self.calls: list[tuple[str, str]] = []

    @property
    def is_ready(self) -> bool:
        return self.ready

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def status(self) -> dict[str, Any]:
        return {
            "model_ready": self.ready,
            "model_name": "fake-model",
            "adapter_loaded": False,
            "device": "cpu",
            "load_error": None if self.ready else "Model has not been loaded.",
        }

    def generate_answer(self, document_markdown: str, question: str) -> GenerationResult:
        if not self.ready:
            raise ModelNotReadyError("not ready")
        self.calls.append((document_markdown, question))
        return GenerationResult(
            answer=self.answer,
            unanswerable=self.unanswerable,
            model_name="fake-model",
            prompt_tokens=10,
            generated_tokens=2,
            document_truncated=False,
            document_chars_total=len(document_markdown),
            document_chars_used=len(document_markdown),
            device="cpu",
            dtype="float32",
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, max_upload_mb=1)


@pytest.fixture
def fake_model() -> FakeModelService:
    return FakeModelService()


@pytest.fixture
def client(settings: Settings, fake_model: FakeModelService) -> TestClient:
    return TestClient(create_app(settings=settings, model_service=fake_model))


> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && cat > /tmp/try_extract.py <<'EOF'
import sys
sys.path.insert(0, 'tests')
from conftest import make_structured_pdf
from app.document_processor import process_pdf
d = process_pdf('a.pdf', make_structured_pdf(), max_bytes=10**7)
for e in d.elements: print(e.page_number, e.type, repr(e.content))
print('-----'); print(d.markdown)
EOF
.venv/bin/python /tmp/try_extract.py"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && PYTHONPATH=. .venv/bin/python /tmp/try_extract.py"

### 🤖 Assistant

> 🔧 called: search_and_replace, execute_command



> 🔧 **search_and_replace** — slm-document-analysis-backend/app/structure.py

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && PYTHONPATH=. .venv/bin/python /tmp/try_extract.py | sed -n 1,8p"

### 🤖 Assistant

> 🔧 called: write_file, write_file, write_file

The extractor now detects the title, headings, bullets and the ruled table. Next come the tests.

> 🔧 **write_file** — """API tests with a fake model (no real model is loaded)."""

from tests.conftest import FakeModelService, make_pdf, make_structured_pdf


def post(client, data=None, filename="report.pdf", question="What was the profit?", content_type="application/pdf"):
    files = {"file": (filename, data if data is not None else make_structured_pdf(), content_type)}
    return client.post("/answer", files=files, data={"question": question})


# --- health ----------------------------------------------------------------------------
def test_health_reflects_model_state(client, fake_model):
    assert client.get("/health").json()["model_ready"] is True
    fake_model.ready = False
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_ready"] is False
    assert body["model_error"]


def test_removed_endpoints_are_gone(client):
    assert client.post("/documents").status_code == 404
    assert client.get("/documents/abc/text").status_code == 404


# --- success -----------------------------------------------------------------------------
def test_single_request_returns_answer_and_structure(client, fake_model):
    fake_model.answer = "30"
    res = post(client)
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "30"
    assert body["question"] == "What was the profit?"
    assert body["unanswerable"] is False
    assert body["model_name"] == "fake-model"
    assert body["inference_time_seconds"] >= 0
    doc = body["document"]
    assert doc["filename"] == "report.pdf"
    assert doc["page_count"] == 2
    assert doc["table_count"] == 1
    assert doc["heading_count"] == 4
    assert body["structured_text"].startswith("# Annual Report 2024")
    assert "| North | 120 | 30 |" in body["structured_text"]
    assert body["structured_text_truncated"] is False
    assert body["disclaimer"]


def test_model_receives_markdown_and_trimmed_question(client, fake_model):
    post(client, question="  What was the profit?  ")
    [(markdown, question)] = fake_model.calls
    assert question == "What was the profit?"
    assert "## 2. Results" in markdown and "| South | 90 | 15 |" in markdown


def test_answer_locations_found_by_string_match(client, fake_model):
    fake_model.answer = "120"
    body = post(client).json()
    [loc] = body["answer_locations"]
    assert loc["page_number"] == 2
    assert loc["element_type"] == "table"
    assert loc["section_path"] == ["Annual Report 2024", "2. Results"] or loc["section_path"][-1] == "2. Results"
    assert loc["table_cell"] == {"value": "120", "row_label": "North", "column_label": "Revenue"}


def test_no_location_when_answer_text_not_in_document(client, fake_model):
    fake_model.answer = "completely unrelated words"
    assert post(client).json()["answer_locations"] == []


def test_unanswerable_is_reported_without_locations(client):
    client.app.state.model_service.answer = "unanswerable"
    client.app.state.model_service.unanswerable = True
    body = post(client).json()
    assert body["unanswerable"] is True
    assert body["answer_locations"] == []
    assert "does not contain" in body["message"]


def test_each_request_is_independent(client, fake_model):
    post(client, data=make_pdf(["Document A talks about apples and oranges."]))
    post(client, data=make_pdf(["Document B talks about trains and planes."]))
    (a, _), (b, _) = fake_model.calls
    assert "apples" in a and "trains" not in a
    assert "trains" in b and "apples" not in b


def test_structured_text_is_bounded(settings, fake_model):
    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.main import create_app

    small = Settings(_env_file=None, max_upload_mb=1, max_markdown_response_chars=1000)
    c = TestClient(create_app(settings=small, model_service=fake_model))
    body = post(c, data=make_pdf(["word " * 800])).json()
    assert len(body["structured_text"]) == 1000
    assert body["structured_text_truncated"] is True


# --- validation errors -----------------------------------------------------------------------
def test_empty_question_rejected(client, fake_model):
    for q in ("", "   "):
        res = post(client, question=q)
        assert res.status_code == 422
        assert res.json()["error"]["code"] in {"empty_question", "validation_error"}
    assert fake_model.calls == []


def test_missing_question_or_file_rejected(client):
    assert client.post("/answer", data={"question": "x"}).status_code == 422
    files = {"file": ("a.pdf", make_structured_pdf(), "application/pdf")}
    assert client.post("/answer", files=files).status_code == 422


def test_overlong_question_rejected(client):
    res = post(client, question="q" * 2001)
    assert res.status_code == 422 and res.json()["error"]["code"] == "question_too_long"


def test_wrong_file_type_rejected(client):
    res = post(client, data=b"hello", filename="notes.txt", content_type="text/plain")
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "unsupported_file_type"


def test_empty_upload_rejected(client):
    res = post(client, data=b"")
    assert res.status_code == 400 and res.json()["error"]["code"] == "empty_file"


def test_oversized_upload_rejected(client):
    res = post(client, data=b"%PDF-1.4\n" + b"0" * (1024 * 1024 + 10))
    assert res.status_code == 413 and res.json()["error"]["code"] == "file_too_large"


def test_malformed_pdf_rejected(client):
    res = post(client, data=b"%PDF-1.4\nthis is garbage, not a pdf")
    assert res.status_code == 422
    assert res.json()["error"]["code"] in {"malformed_pdf", "no_extractable_text"}


def test_pdf_without_text_rejected(client):
    res = post(client, data=make_pdf(["", ""]))
    assert res.status_code == 422 and res.json()["error"]["code"] == "no_extractable_text"


# --- model state / failures --------------------------------------------------------------------
def test_model_not_ready_returns_503_without_processing(client, fake_model):
    fake_model.ready = False
    res = post(client)
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "model_not_ready"


def test_inference_failure_returns_500_with_generic_message(settings):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.model_service import InferenceError

    class Failing(FakeModelService):
        def generate_answer(self, document_markdown, question):
            raise InferenceError("Generation failed (RuntimeError).")

    res = post(TestClient(create_app(settings=settings, model_service=Failing())))
    assert res.status_code == 500
    assert res.json()["error"] == {"code": "inference_failed", "message": "Generation failed (RuntimeError)."}


def test_inference_timeout_returns_504(settings):
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.model_service import InferenceTimeoutError

    class Slow(FakeModelService):
        def generate_answer(self, document_markdown, question):
            raise InferenceTimeoutError("Answer generation exceeded 120 seconds.")

    res = post(TestClient(create_app(settings=settings, model_service=Slow())))
    assert res.status_code == 504 and res.json()["error"]["code"] == "inference_timeout"


def test_unexpected_error_is_generic(settings):
    from fastapi.testclient import TestClient
    from app.main import create_app

    class Boom(FakeModelService):
        def generate_answer(self, document_markdown, question):
            raise ZeroDivisionError("secret /home/user/path")

    res = TestClient(create_app(settings=settings, model_service=Boom()), raise_server_exceptions=False)
    r = post(res)
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "An unexpected error occurred."}}


# --- CORS ----------------------------------------------------------------------------------------
def test_cors_allows_configured_origin_only(client):
    ok = client.options(
        "/answer",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options(
        "/answer",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers


> 🔧 **write_file** — """Tests for PDF validation and structure extraction (no model needed)."""

import pymupdf
import pytest

from app.document_processor import (
    EmptyFileError,
    FileTooLargeError,
    MalformedPDFError,
    NoExtractableTextError,
    TooManyPagesError,
    UnsupportedFileTypeError,
    process_pdf,
    sanitize_filename,
    validate_pdf_upload,
)
from tests.conftest import make_pdf, make_structured_pdf

MAX = 5 * 1024 * 1024


def _types(doc):
    return [e.type for e in doc.elements]


# --- structure ----------------------------------------------------------------------
def test_structured_pdf_headings_lists_and_table_are_recovered():
    doc = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX)
    assert doc.page_count == 2
    assert _types(doc) == [
        "title", "heading_1", "paragraph", "heading_2", "list_item", "list_item",
        "heading_1", "table", "paragraph",
    ]
    assert doc.heading_count == 4
    assert doc.table_count == 1
    assert [e.page_number for e in doc.elements] == [1, 1, 1, 1, 1, 1, 2, 2, 2]


def test_paragraph_lines_are_joined_into_one_paragraph():
    doc = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX)
    paragraph = next(e for e in doc.elements if e.type == "paragraph")
    assert paragraph.content.startswith("This report describes")
    assert "Revenue increased" in paragraph.content
    assert "\n" not in paragraph.content


def test_markdown_has_heading_markers_and_markdown_table():
    md = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX).markdown
    assert md.startswith("# Annual Report 2024")
    assert "\n## 1. Introduction\n" in md
    assert "\n### 1.1 Scope\n" in md
    assert "- Financial results\n- Customer satisfaction" in md
    assert "| Region | Revenue | Profit |\n| --- | --- | --- |\n| North | 120 | 30 |" in md


def test_table_text_is_not_repeated_as_paragraph():
    doc = process_pdf("report.pdf", make_structured_pdf(), max_bytes=MAX)
    paragraphs = " ".join(str(e.content) for e in doc.elements if e.type == "paragraph")
    assert "Revenue" not in paragraphs.replace("Revenue increased", "")
    assert "120" not in paragraphs


def test_plain_pdf_without_styling_yields_paragraphs_only():
    doc = process_pdf("plain.pdf", make_pdf(["Just a plain sentence about nothing special."]), max_bytes=MAX)
    assert _types(doc) == ["paragraph"]
    assert doc.heading_count == 0 and doc.table_count == 0


def test_page_numbers_follow_the_pdf():
    doc = process_pdf(
        "two.pdf",
        make_pdf(["First page has enough words here.", "Second page has other words here."]),
        max_bytes=MAX,
    )
    assert [e.page_number for e in doc.elements] == [1, 2]


def test_repeated_page_numbers_in_footer_are_dropped():
    pdf = pymupdf.open()
    for n in range(1, 4):
        page = pdf.new_page()
        page.insert_text((50, 100), f"Body text of page number {n} with enough words.", fontsize=11)
        page.insert_text((290, 820), str(n), fontsize=9)
    data = pdf.tobytes()
    doc = process_pdf("n.pdf", data, max_bytes=MAX)
    assert all(not str(e.content).strip().isdigit() for e in doc.elements)


def test_blank_page_is_reported_but_document_is_accepted():
    pdf = make_pdf(["Content on the first page, long enough.", ""])
    doc = process_pdf("blank.pdf", pdf, max_bytes=MAX)
    assert doc.empty_page_numbers == [2]


# --- rejections ------------------------------------------------------------------------
def test_wrong_extension_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        process_pdf("notes.txt", b"%PDF-1.4 hello", max_bytes=MAX)


def test_fake_pdf_without_signature_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        process_pdf("fake.pdf", b"this is not a pdf at all", max_bytes=MAX)


def test_empty_file_is_rejected():
    with pytest.raises(EmptyFileError):
        process_pdf("empty.pdf", b"", max_bytes=MAX)


def test_oversized_file_is_rejected():
    with pytest.raises(FileTooLargeError):
        process_pdf("big.pdf", make_pdf(["Some text that is long enough."]), max_bytes=100)


def test_truncated_pdf_is_malformed_or_unreadable():
    data = make_pdf(["Some text that is long enough to matter."])[:60]
    with pytest.raises((MalformedPDFError, NoExtractableTextError)):
        process_pdf("broken.pdf", data, max_bytes=MAX)


def test_pdf_without_text_is_rejected():
    with pytest.raises(NoExtractableTextError):
        process_pdf("blank.pdf", make_pdf(["", ""]), max_bytes=MAX)


def test_too_many_pages_is_rejected():
    with pytest.raises(TooManyPagesError):
        process_pdf("long.pdf", make_pdf(["Some text on this page."] * 3), max_bytes=MAX, max_pages=2)


def test_password_protected_pdf_is_rejected():
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 100), "Secret text that is long enough.")
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
    )
    with pytest.raises(MalformedPDFError):
        process_pdf("locked.pdf", data, max_bytes=MAX)


# --- helpers ---------------------------------------------------------------------------
def test_validate_accepts_valid_pdf():
    validate_pdf_upload("a.pdf", make_pdf(["hello world"]), MAX)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("", "document.pdf"),
        (None, "document.pdf"),
        ("a\x00b.pdf", "ab.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


> 🔧 **write_file** — """Tests for Markdown rendering, answer location and the training-format prompt."""

from app.prompting import build_prompt, first_line, is_unanswerable, prompt_parts
from app.structure import (
    DocumentElement,
    elements_to_markdown,
    locate_answer,
    normalize,
    table_to_markdown,
)

TABLE = [["Region", "Revenue", "Profit"], ["North", "120", "30"], ["South", "90", "15"]]


def elements():
    return [
        DocumentElement("title", "Annual Report", 1),
        DocumentElement("heading_1", "1. Introduction", 1),
        DocumentElement("paragraph", "The company grew in the northern region.", 1),
        DocumentElement("heading_1", "2. Results", 2),
        DocumentElement("heading_2", "2.1 Regional table", 2),
        DocumentElement("table", TABLE, 2),
        DocumentElement("list_item", "• Hire 21 engineers", 2),
    ]


# --- Markdown -----------------------------------------------------------------------
def test_table_markdown_has_header_separator_and_rows():
    assert table_to_markdown(TABLE).splitlines() == [
        "| Region | Revenue | Profit |",
        "| --- | --- | --- |",
        "| North | 120 | 30 |",
        "| South | 90 | 15 |",
    ]


def test_table_markdown_escapes_pipes_and_pads_short_rows():
    out = table_to_markdown([["A", "B"], ["x|y"]]).splitlines()
    assert out[2] == "| x\\|y |  |"


def test_markdown_heading_levels_and_lists():
    md = elements_to_markdown(elements())
    assert "# Annual Report\n\n## 1. Introduction" in md
    assert "### 2.1 Regional table" in md
    assert md.endswith("- Hire 21 engineers")


def test_consecutive_list_items_stay_adjacent():
    md = elements_to_markdown(
        [DocumentElement("list_item", "- a", 1), DocumentElement("list_item", "- b", 1)]
    )
    assert md == "- a\n- b"


def test_numbered_list_items_keep_their_numbers():
    md = elements_to_markdown([DocumentElement("list_item", "1. first", 1)])
    assert md == "1. first"


# --- locate_answer ----------------------------------------------------------------------
def test_locate_in_paragraph_returns_page_and_section():
    [loc] = locate_answer(elements(), "northern region")
    assert loc.page_number == 1
    assert loc.element_type == "paragraph"
    assert loc.section_path == ("Annual Report", "1. Introduction")
    assert loc.table_cell is None


def test_locate_in_table_returns_row_and_column_labels():
    [loc] = locate_answer(elements(), "120")
    assert loc.page_number == 2
    assert loc.element_type == "table"
    assert loc.section_path == ("Annual Report", "2. Results", "2.1 Regional table")
    assert loc.table_cell.value == "120"
    assert loc.table_cell.row_label == "North"
    assert loc.table_cell.column_label == "Revenue"


def test_locate_matches_whole_words_only():
    # "2" must not match inside "21" or "120"
    assert locate_answer(elements(), "2") == []


def test_locate_ignores_case_and_punctuation():
    assert len(locate_answer(elements(), "The Company grew!")) == 1


def test_locate_returns_nothing_when_answer_absent():
    assert locate_answer(elements(), "Zubrowka") == []
    assert locate_answer(elements(), "") == []
    assert locate_answer(elements(), "...") == []


def test_locate_respects_max_results():
    els = [DocumentElement("paragraph", "alpha beta", n) for n in range(1, 10)]
    assert len(locate_answer(els, "alpha", max_results=3)) == 3


def test_normalize():
    assert normalize("  Hello,   World! ") == "hello world"


# --- prompt format (must equal the training notebook) ---------------------------------------
def test_prompt_is_exactly_the_training_format():
    # Reproduction of ``prompt(ex)`` from notebooks/simple_pipeline.ipynb, Step 3.
    def training_prompt(ex):
        return f"### Context:\n{ex['context'].strip()}\n\n### Question:\n{ex['question'].strip()}\n\n### Answer:\n"

    ctx, q = "  # Title\n\nSome text.  ", "  What is it?  "
    assert build_prompt(ctx, q) == training_prompt({"context": ctx, "question": q})
    assert build_prompt(ctx, q).endswith("### Answer:\n")


def test_prompt_parts_recombine_to_the_full_prompt():
    prefix, suffix = prompt_parts("What?")
    assert prefix + "DOC" + suffix == build_prompt("DOC", "What?")


def test_first_line_and_unanswerable():
    assert first_line("  120\nextra text") == "120"
    assert first_line("") == ""
    assert is_unanswerable("unanswerable")
    assert is_unanswerable(" Unanswerable. ")
    assert not is_unanswerable("The answer is unanswerable")


### 🤖 Assistant

> 🔧 called: write_file, write_file, execute_command

The API test file didn't get written, because the path was missing again. I'll redo it, and also tidy the one clumsy assertion on `section_path`.

> 🔧 **write_file** — slm-document-analysis-backend/tests/test_api.py

> 🔧 **write_file** — slm-document-analysis-backend/tests/test_model_integration.py

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && cat pytest.ini && .venv/bin/python -m pytest -q 2>&1 | tail -40"

### 🤖 Assistant

> 🔧 called: search_and_replace, execute_command

That failure is a flaw in the test, not in the code: "2" is legitimately a whole word in the heading "2. Results". I'll test the intended behaviour instead, that "12" must not match inside "120".

> 🔧 **search_and_replace** — slm-document-analysis-backend/tests/test_structure.py

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && .venv/bin/python -m pytest -q 2>&1 | tail -8"

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — slm-document-analysis-backend/tests/test_structure.py

### 🤖 Assistant

> 🔧 called: apply_diff, execute_command



> 🔧 **apply_diff** — slm-document-analysis-backend/tests/test_structure.py

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && .venv/bin/python -m pytest -q 2>&1 | tail -8"

### 🤖 Assistant

> 🔧 called: str_replace_placeholder

One of those assertions is needlessly convoluted, so I'll simplify it.

> 🔧 **str_replace_placeholder** — done

### 🤖 Assistant

> 🔧 called: apply_diff



> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:86
-------
    # a partial word does not match
    assert locate_answer(elements(), "north") == [] or all(
        l.element_type != "table" or l.table_cell.value.lower() == "north"
        for l in locate_answer(elements(), "north")
    )
    assert locate_answer(elements(), "nor") == []
=======
    # a partial word does not match ("nor" is inside "northern"/"North")
    assert locate_answer(elements(), "nor") == []
>>>>>>> REPLACE

### 🤖 Assistant

> 🔧 called: execute_command

Now I'll check the model path with the tiny test model. That covers prompt-id equality, the first-line answer, and truncation that cuts only the document.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && MODEL_NAME_OR_PATH=hf-internal-testing/tiny-random-LlamaForCausalLM MODEL_DEVICE=cpu .venv/bin/python -m pytest -q -m requires_model tests/test_model_integration.py 2>&1 | tail -15; .venv/bin/python -m pytest -q 2>&1 | tail -3"

### 🤖 Assistant

> 🔧 called: execute_command

The mechanics tests pass on the tiny model. Next, a sanity run with the real `Qwen/Qwen1.5-0.5B` base on CPU. There's no adapter locally, so this tests the pipeline and not answer quality.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && cat > /tmp/real_e2e.py <<'EOF'
import sys, time
sys.path.insert(0, 'tests')
from fastapi.testclient import TestClient
from app.config import Settings
from app.main import create_app
from conftest import make_structured_pdf
s = Settings(_env_file=None, model_name_or_path='Qwen/Qwen1.5-0.5B', model_device='cpu', max_new_tokens=16)
t=time.time()
with TestClient(create_app(settings=s)) as c:
    print('load s', round(time.time()-t,1), c.get('/health').json())
    r = c.post('/answer', files={'file':('r.pdf', make_structured_pdf(), 'application/pdf')}, data={'question':'What was the profit of the North region?'})
    print(r.status_code)
    b = r.json()
    print({k:b[k] for k in ('answer','unanswerable','model_name','inference_time_seconds','prompt_tokens','generated_tokens','document_truncated','answer_locations')} if r.status_code==200 else b)
EOF
PYTHONPATH=. timeout 250 .venv/bin/python /tmp/real_e2e.py 2>&1 | grep -v Warning | tail -15"

### 🤖 Assistant

> 🔧 called: update_todo_list, execute_command



> 🔧 **update_todo_list** — [x] Inspect environment, existing files and the training notebook
[x] Scaffold React+TS+Vite frontend (old two-step UI)
[x] Add GET /documents/{id}/text (old flow)
[x] Create colab/ builder script
[x] Generate slm_document_analysis_backend.ipynb (old flow)
[x] Verify notebook locally (old flow)
[x] Backend: structure.py, structure-aware document_processor.py
[x] Backend: training-format prompting.py, model_service.py (first line, greedy, document-only truncation)
[x] Backend: single POST /answer in main.py, schemas.py, config.py; delete chunking/retrieval/storage
[x] Backend tests rewritten (59 passed, 3 model tests skipped by default) and verified with tiny model + real Qwen1.5-0.5B base on CPU
[-] Backend: .env.example, requirements.txt, smoke_test.sh, backend README
[ ] Colab: update build_notebook.py + verify_notebook.py, rebuild and re-verify
[ ] Frontend: single-submit UI, types/api/tests, build + test
[ ] Root README + final cleanup (dist, temp files) + final summary

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && cat .env.example requirements.txt scripts/smoke_test.sh && wc -l README.md"

### 🤖 Assistant

> 🔧 called: write_file, write_file, execute_command

The todo tool needs the exact wording of earlier completed items, so I'll leave it alone and carry on. Next are the config files and the smoke script.

> 🔧 **write_file** — slm-document-analysis-backend/.env.example

> 🔧 **write_file** — slm-document-analysis-backend/scripts/smoke_test.sh

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && cat .gitignore | head -20; sed -n 1,243p README.md"

### 🤖 Assistant

> 🔧 called: write_file, execute_command

I'm rewriting the backend README so it describes the new flow.

> 🔧 **write_file** — # SLM Document Analysis Backend

FastAPI backend for the M.Tech project *"A Study of Parameter-Efficient Fine-Tuning of Small
Language Models (SLMs) for Document Analysis"*.

It performs **inference only**. The browser sends **one request containing a PDF and a question**.
The backend extracts the PDF's text **with its structure kept** (headings, lists, tables), puts the
whole document and the question into the **same prompt format used for fine-tuning**, runs the
Small Language Model (base model + LoRA adapter), and returns the answer. It does **not** train
anything, and there is **no chunking, no retrieval/RAG, and no server-side storage**.

```
Frontend --(PDF + question, one multipart request)--> POST /answer
   -> structure-preserving extraction (PyMuPDF)  -> Markdown
   -> "### Context:\n{markdown}\n\n### Question:\n{q}\n\n### Answer:\n"
   -> Qwen base model + LoRA adapter (greedy decoding)  -> first line = answer
   -> JSON response
```

## Project structure

```
slm-document-analysis-backend/
├── app/
│   ├── main.py                # FastAPI app: GET /health, POST /answer, error handlers, CORS
│   ├── config.py              # Settings read from environment variables / .env
│   ├── errors.py              # ApiError (uniform error responses)
│   ├── schemas.py             # Pydantic response models
│   ├── structure.py           # DocumentElement model, Markdown rendering, locate_answer()
│   ├── document_processor.py  # PDF validation + structure extraction (PyMuPDF heuristics)
│   ├── prompting.py           # The training prompt template (must match training)
│   └── model_service.py       # Model + LoRA loading, token budget, generation
├── tests/
│   ├── conftest.py            # PDF generators + fake model service
│   ├── test_document_processor.py
│   ├── test_structure.py      # Markdown, locate_answer, exact prompt format
│   ├── test_api.py
│   └── test_model_integration.py   # needs a real model; skipped by default
├── scripts/smoke_test.sh      # manual end-to-end check with curl
├── .env.example  .gitignore  pytest.ini  requirements.txt  README.md
```

## 1. Set up (macOS)

Requires Python 3.11–3.13 (any version supported by your PyTorch/Transformers install).

```bash
cd slm-document-analysis-backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## 2. Configure your model (REQUIRED)

No model location is assumed. Edit `.env`:

| Variable | What to put there |
|---|---|
| `MODEL_NAME_OR_PATH` | The **same base model** you fine-tuned: a Hugging Face id or a local folder. Your notebook `simple_pipeline.ipynb` used `Qwen/Qwen1.5-0.5B`; set it only if that is what produced your adapter. |
| `ADAPTER_PATH` | The folder the notebook saved with `model.save_pretrained(f"{OUT}/adapter")` (contains `adapter_config.json`). If empty, the **un-fine-tuned base model** runs, which was never trained on the `### Answer:` format. |
| `MODEL_DEVICE` | `auto` (cuda → mps → cpu), `cpu`, `mps`, `cuda`. |
| `MODEL_DTYPE` | `auto` (float32 on CPU, float16 otherwise), or `float32` / `float16` / `bfloat16`. |
| `MAX_NEW_TOKENS` | Default 64 (the notebook evaluated with 32). |
| `MAX_INPUT_TOKENS` | Default 8192; upper limit for document + question tokens. |

If `MODEL_NAME_OR_PATH` is empty the server still starts, `/health` returns `model_ready: false` with
the reason, and `/answer` returns HTTP 503 `model_not_ready`. It never substitutes another model.

### Prompt format (matches the training notebook)

`app/prompting.py` builds exactly what `prompt(ex)` in `simple_pipeline.ipynb` (Step 3) builds:

```
### Context:
{document as Markdown}

### Question:
{question}

### Answer:
```

No instruction sentence and no chat template are added, because the adapter never saw them. The
model's output is decoded from the **new tokens only**, with **greedy decoding**, and the **first
line** is the answer (this is how the notebook's evaluation read it). The training label
`unanswerable` is recognised and reported as `unanswerable: true`.

### What "structure-preserving" means

| PDF content | Markdown given to the model |
|---|---|
| Title / headings | `#`, `##`, `###`, `####` (from font size/bold and section numbers like `1.1`) |
| Ruled tables (`page.find_tables()`) | `\| col \| col \|` rows with a `\| --- \|` separator |
| Bullets / numbered items | `- item` / `1. item` |
| Body text | paragraphs separated by blank lines (wrapped lines joined) |
| Page numbers, repeated headers/footers | removed |

Heuristics, not magic: headings are recognised by size/bold, so documents that style headings
differently may be mis-classified; tables **without ruling lines** are read as ordinary text; multi-column
pages may be read in the wrong order; **no OCR** (scanned PDFs are rejected).

### Long documents

The whole document is sent. If document + question exceed `MAX_INPUT_TOKENS` (or the model's context
window), the **end of the document** is cut (never the question or `### Answer:`). The response then has
`document_truncated: true`, `document_chars_used` and `document_chars_total`. Note that your adapter was trained
with `MAX_LEN = 384` tokens, so quality on documents of several pages is **not established by the training
set** — evaluate it before relying on it.

## 3. Run

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: <http://127.0.0.1:8000/docs>. Quick manual check:

```bash
scripts/smoke_test.sh path/to/file.pdf "What is this document about?"
```

## 4. Run the tests

```bash
pytest                          # unit + API tests; no model download needed
MODEL_NAME_OR_PATH=<model> [ADAPTER_PATH=<dir>] pytest -m requires_model   # mechanics with a real model
```

The API tests use a **fake model service**; they verify HTTP behaviour and the prompt/Markdown plumbing,
not real model output. The `requires_model` tests check mechanics (prompt token ids, first-line answer,
document-only truncation); they do not measure answer quality.

## 5. API reference

All errors use one shape: `{"error": {"code": "<machine_code>", "message": "<text>"}}`.

### `GET /health`
```json
{"status":"ok","model_ready":true,"model_name":"Qwen/Qwen1.5-0.5B + LoRA adapter 'adapter'",
 "adapter_loaded":true,"device":"cuda","model_error":null}
```

### `POST /answer` (multipart/form-data: `file` = PDF, `question` = text)

```bash
curl -F "file=@report.pdf;type=application/pdf" -F "question=What was the profit of the North region?" \
  http://127.0.0.1:8000/answer
```

Response 200:
```json
{
  "question": "What was the profit of the North region?",
  "answer": "30",
  "unanswerable": false,
  "message": null,
  "model_name": "Qwen/Qwen1.5-0.5B + LoRA adapter 'adapter'",
  "inference_time_seconds": 1.25,
  "prompt_tokens": 122,
  "generated_tokens": 3,
  "document": {"filename":"report.pdf","page_count":2,"heading_count":4,"table_count":1,
               "character_count":388,"word_count":70,"empty_page_numbers":[]},
  "document_truncated": false,
  "document_chars_used": 388,
  "document_chars_total": 388,
  "structured_text": "# Annual Report 2024\n\n## 1. Introduction\n\n…| Region | Revenue | Profit |\n| --- | --- | --- |\n| North | 120 | 30 |…",
  "structured_text_truncated": false,
  "answer_locations": [
    {"page_number":2,"element_type":"table","section_path":["Annual Report 2024","2. Results"],
     "snippet":"North | 120 | 30","table_cell":{"value":"30","row_label":"North","column_label":"Profit"}}
  ],
  "disclaimer": "Answers are generated by a small language model and may be incorrect. Check them against the document."
}
```
(Illustrative values; every field is computed per request.)

* `answer` is the first line of the model's output. `unanswerable` is true when it equals the training label
  `unanswerable`; `message` then explains it.
* `structured_text` is the exact Markdown (bounded by `MAX_MARKDOWN_RESPONSE_CHARS`) that was placed in the
  prompt, so the user can see what the model was given.
* `answer_locations` is **not produced by the model**. After generation the backend searches the extracted
  document for the answer text (case/punctuation-insensitive, whole words). Each hit reports the page,
  the heading path, and for tables the row and column labels. It is `[]` when the answer text does not
  appear verbatim (for example when the model rephrases or is wrong), and always `[]` for `unanswerable`.
  Short answers such as `2` can match several places; up to 5 are returned.
* No confidence score is returned because the model does not provide a calibrated one.

| Status | code | Cause |
|---|---|---|
| 400 | `empty_file` | zero-byte upload |
| 413 | `file_too_large` / `too_many_pages` | over `MAX_UPLOAD_MB` / `MAX_PAGES` |
| 415 | `unsupported_file_type` | not a `.pdf` or no PDF signature |
| 422 | `malformed_pdf` | corrupt / password-protected PDF |
| 422 | `no_extractable_text` | no meaningful text (e.g. scanned PDF; OCR not supported) |
| 422 | `empty_question` / `question_too_long` / `validation_error` | blank, >`MAX_QUESTION_CHARS`, or a missing field |
| 503 | `model_not_ready` / `model_not_configured` | model not loaded (checked before the PDF is processed) |
| 504 | `inference_timeout` | exceeded `GENERATION_TIMEOUT_SECONDS` |
| 500 | `inference_failed` / `internal_error` / `processing_failed` | other failures (details only in server logs) |

## 6. Frontend example (React / Vite, origin `http://localhost:5173`)

```js
const API = "http://127.0.0.1:8000";

export async function health() {
  return (await fetch(`${API}/health`)).json();
}

export async function ask(file, question) {
  const form = new FormData();
  form.append("file", file);          // field names must be "file" and "question"
  form.append("question", question);  // do not set Content-Type manually
  const res = await fetch(`${API}/answer`, { method: "POST", body: form });
  const body = await res.json();
  if (!res.ok) throw new Error(body.error?.message ?? "Request failed");
  return body;
}
```

Other origins must be added to `CORS_ORIGINS`. Each question re-sends the PDF, because the server
keeps nothing between requests.

## 7. Design notes

* **Stateless:** nothing is stored. The PDF is read into memory for the request and discarded. The
  filename is only a display label and is never used as a path. Run with any number of workers; each
  keeps its own copy of the model, so one worker is advisable for large models.
* **One generation at a time:** a lock serialises generation; concurrent requests queue.
* **Security:** CORS is restricted to `CORS_ORIGINS`; the server binds `127.0.0.1`; errors never include
  stack traces or filesystem paths; document text is never logged (only counts).

## 8. Can the model run locally?

I could not inspect your adapter, but the notebook shows a **0.5B-parameter** base, which is small.

* `Qwen/Qwen1.5-0.5B` in float32 on **CPU** was run here (without your adapter) at about 3.8 s for a
  122-token prompt and 16 new tokens. Long (several-page) prompts are slower on CPU.
* **MacBook Air (Apple Silicon):** `MODEL_DEVICE=mps` or `cpu` should work for a 0.5B model; memory is
  shared, so a very long prompt (`MAX_INPUT_TOKENS`) needs more RAM. **This was not tested on a Mac.**
* **Google Colab (GPU):** see `../colab/` — the same backend runs there and is reached through a tunnel.
* **Remote inference from a local backend (NOT implemented).** A class with the same interface as
  `ModelService` (`is_ready`, `status`, `generate_answer`) could be passed to `create_app(model_service=...)`.

## 9. Known limitations

* No OCR; structure detection is heuristic (see above).
* Quality on multi-page documents has not been measured; the adapter was trained on ≤384-token examples.
* No authentication, no persistence, no streaming responses.


> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && wc -l build_notebook.py verify_notebook.py && cat requirements.txt"

### 🤖 Assistant

> 🔧 called: read_file, read_file

Now the Colab notebook builder.

> 🔧 **read_file** — colab/verify_notebook.py

> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/slm-document-analysis-backend && sed -i 's/\bAnswerLocation\b/AnswerMatch/g' app/structure.py && grep -n 'AnswerMatch\|AnswerLocation' app/structure.py app/main.py app/schemas.py tests/*.py | head -20"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && sed -i '595,622d;347,388d;251,345d' build_notebook.py && sed -n 245,256p build_notebook.py && grep -n 'build_cells\|^CELL' build_notebook.py"

### 🤖 Assistant

> 🔧 called: execute_command

The line-range deletion removed the old cells 4–10 and `build_cells`, as planned. Now I'll rewrite the intro, Cell 2 and the build function for the single-request design.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && cat > /tmp/patch_nb.py <<'PYEOF'
import re
p = 'build_notebook.py'
s = open(p, encoding='utf-8').read()

# ---------- intro ----------
a = s.index('INTRO_MD = \"\"\"')
b = s.index('CELL1_MD = \"\"\"')
intro = '''INTRO_MD = \"\"\"\\
# SLM Document Analysis: Backend (Google Colab)

This notebook runs the **backend** for the *SLM Document Analysis* proof of concept. The React
frontend sends **one request containing a PDF and a question**. The backend extracts the PDF text
**with its structure kept** (headings, lists, tables as Markdown), builds a prompt in the **same format
the model was fine-tuned with**, runs your **Qwen base model + LoRA adapter**, and returns the answer.
The frontend reaches it over HTTPS through a temporary tunnel.

**Inference only.** Nothing is trained here. The LoRA adapter is an already-trained artefact that is
only *loaded*. There is **no chunking, no retrieval (no RAG) and no server-side storage**: the whole
document goes into the prompt and nothing is kept after the response.

**How to run**
1. `Runtime -> Change runtime type -> GPU` (a T4 is plenty for a 0.5B model).
2. Fill in **Cell 2** (model and adapter). No model is assumed: you must supply it.
3. Run the cells **in order**, 1 to 8. Cell 8 prints the URL that goes into the frontend's `.env`.

**Limitations (read this)**
* Text-based PDFs only. **There is no OCR**: scanned PDFs are rejected.
* Structure detection is heuristic (font size / bold / ruling lines). Tables without ruling lines are
  read as plain text.
* Your adapter was trained on examples of at most 384 tokens. Whole multi-page documents are far
  longer, so answer quality on them has to be evaluated; it is not established by the training set.
* If a document exceeds `MAX_INPUT_TOKENS`, the END of the document is cut and the response says so.
* The tunnel URL changes every time you start it. Update the frontend `.env` and restart `npm run dev`.
* Answers come from a small model and can be wrong.

> The Python in this notebook is generated from the unit-tested backend modules in
> `slm-document-analysis-backend/app/` by `colab/build_notebook.py`. If you change the backend
> source, rebuild the notebook instead of editing both.
\"\"\"

'''
s = s[:a] + intro + s[b:]

# ---------- cell 1 package list text ----------
s = s.replace('CELL1_MD = \"\"\"\\\\\\n## Cell 1', 'CELL1_MD = \"\"\"\\\\\\n## Cell 1')

# ---------- cell 2 user block ----------
a = s.index('# --- Chunking and retrieval')
b = s.index('# --- Uploads')
gen = '''# --- Generation (greedy decoding, as in the training notebook's evaluation) ----------------
MAX_NEW_TOKENS = 64          # the training notebook evaluated with 32; the answer is the first line
MAX_INPUT_TOKENS = 8192      # document + question tokens; the END of a longer document is cut
# Keep this BELOW ~100 s: Cloudflare tunnels cut requests off at about 100 s with an HTML error page.
GENERATION_TIMEOUT_SECONDS = 90

'''
s = s[:a] + gen + s[b:]
s = s.replace('MAX_UPLOAD_MB = 20\\n\\n# --- API server', 'MAX_UPLOAD_MB = 20\\nMAX_PAGES = 100\\n\\n# --- API server')

# ---------- build settings ----------
s = s.replace('''    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    retrieval_top_k=RETRIEVAL_TOP_K,
    max_context_chars=MAX_CONTEXT_CHARS,
''', '')
s = s.replace('    max_upload_mb=MAX_UPLOAD_MB,\\n', '    max_upload_mb=MAX_UPLOAD_MB,\\n    max_pages=MAX_PAGES,\\n')
s = s.replace('print(\"  chunks  :\", settings.chunk_size, \"chars, overlap\", settings.chunk_overlap)\\n', 'print(\"  limits  :\", settings.max_input_tokens, \"prompt tokens,\", settings.max_new_tokens, \"new tokens,\", settings.max_pages, \"pages\")\\n')
open(p, 'w', encoding='utf-8').write(s)
PYEOF
python3 /tmp/patch_nb.py && grep -n 'CHUNK\|chunk\|RETRIEVAL\|retriev' build_notebook.py"

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: read_file

The shell patch failed on PowerShell quoting, so nothing was applied. I'll edit with the file tools instead, starting with the tail of the file.

> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: apply_diff



> 🔧 **apply_diff** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: execute_command

I need to rename the old Cell 11 and 11b to 8 and 8b, and update the intro, Cell 2 and the CORS check. I'll make those edits with the file tools.

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && sed -i 's/CELL11B_MD/CELL8B_MD/g; s/CELL11B_CODE/CELL8B_CODE/g; s/CELL11_MD/CELL8_MD/g; s/CELL11_CODE/CELL8_CODE/g; s/## Cell 11: Start/## Cell 8: Start/; s/### Cell 11b:/### Cell 8b:/; s/(Cell 11b)/(Cell 8b)/; s/re-run Cell 11\./re-run Cell 8./; s/Cells 2 to 10/Cells 2 to 7/; s#{LOCAL_URL}/documents#{LOCAL_URL}/answer#' build_notebook.py && grep -n 'Cell 1[01]\|CELL1[01]\|/documents' build_notebook.py"

### 🤖 Assistant

> 🔧 called: read_file

Next the intro text and the Cell 2 user block.

> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: apply_diff, execute_command



> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:54
-------
This notebook runs the **backend** for the *SLM Document Analysis* proof of concept:
PDF text extraction, chunking, retrieval, and question answering with your **Small Language Model
(base model + optional LoRA adapter)**. The React frontend on your MacBook talks to it over HTTPS
through a temporary tunnel.

**Inference only.** Uploading a PDF never trains or fine-tunes anything. The LoRA adapter is a
separate, already-trained artefact that is only *loaded* here.

**How to run**
1. `Runtime -> Change runtime type -> GPU` (a T4 is enough for models of up to a few billion parameters).
2. Fill in **Cell 2** (model and adapter). No model is assumed: you must supply it.
3. Run the cells **in order**, 1 to 11. Cell 11 prints the URL that goes into the frontend's `.env`.

**Limitations (read this)**
* Documents are stored **in memory only**. They are lost when the Colab session restarts or is
  disconnected, and nothing is written to disk.
* The tunnel URL changes every time you start it. Update the frontend `.env` and restart `npm run dev`.
* Text-based PDFs only. **There is no OCR**: scanned PDFs are rejected.
* Retrieval is lexical (BM25). It is **not** semantic search.
* Answers come from a small model and can be wrong.
=======
This notebook runs the **backend** for the *SLM Document Analysis* proof of concept. The React
frontend sends **one request containing a PDF and a question**. The backend extracts the PDF text
**with its structure kept** (headings, lists, tables as Markdown), builds a prompt in the **same format
the model was fine-tuned with**, runs your **Small Language Model (base model + LoRA adapter)**, and
returns the answer. The frontend on your MacBook reaches it over HTTPS through a temporary tunnel.

**Inference only.** Nothing is trained here. The LoRA adapter is an already-trained artefact that is
only *loaded*. There is **no chunking, no retrieval (no RAG) and no server-side storage**: the whole
document goes into the prompt and nothing is kept after the response is sent.

**How to run**
1. `Runtime -> Change runtime type -> GPU` (a T4 is plenty for a 0.5B model).
2. Fill in **Cell 2** (model and adapter). No model is assumed: you must supply it.
3. Run the cells **in order**, 1 to 8. Cell 8 prints the URL that goes into the frontend's `.env`.

**Limitations (read this)**
* Text-based PDFs only. **There is no OCR**: scanned PDFs are rejected.
* Structure detection is heuristic (font size, bold, ruling lines). Tables without ruling lines are
  read as plain text.
* The adapter was trained on examples of at most 384 tokens. Whole multi-page documents are far
  longer, so answer quality on them must be evaluated; the training set does not establish it.
* If a document exceeds `MAX_INPUT_TOKENS`, the END of the document is cut and the response says so.
* The tunnel URL changes every time you start it. Update the frontend `.env` and restart `npm run dev`.
* Answers come from a small model and can be wrong.
>>>>>>> REPLACE
<<<<<<< SEARCH
:start_line:142
-------
# --- Chunking and retrieval -----------------------------------------------------------------
CHUNK_SIZE = 1000            # characters per chunk
CHUNK_OVERLAP = 150          # characters shared between consecutive chunks (must be < CHUNK_SIZE)
RETRIEVAL_TOP_K = 4          # maximum number of chunks given to the model
MAX_CONTEXT_CHARS = 6000     # maximum total characters of retrieved context

# --- Generation -----------------------------------------------------------------------------
MAX_NEW_TOKENS = 256         # maximum answer length in tokens
MAX_INPUT_TOKENS = 3072      # prompt budget (also capped by the model's context window)
# Keep this BELOW ~100 s: Cloudflare tunnels cut requests off at about 100 s with an HTML error page.
GENERATION_TIMEOUT_SECONDS = 90

# --- Uploads -------------------------------------------------------------------------------
MAX_UPLOAD_MB = 20
=======
# --- Generation (greedy decoding, as in the training notebook's evaluation) -------------------
MAX_NEW_TOKENS = 64          # the training notebook evaluated with 32; the answer is the first line
MAX_INPUT_TOKENS = 8192      # document + question tokens; the END of a longer document is cut
# Keep this BELOW ~100 s: Cloudflare tunnels cut requests off at about 100 s with an HTML error page.
GENERATION_TIMEOUT_SECONDS = 90

# --- Uploads -------------------------------------------------------------------------------
MAX_UPLOAD_MB = 20
MAX_PAGES = 100
>>>>>>> REPLACE
<<<<<<< SEARCH
:start_line:180
-------
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    retrieval_top_k=RETRIEVAL_TOP_K,
    max_context_chars=MAX_CONTEXT_CHARS,
    max_new_tokens=MAX_NEW_TOKENS,
    max_input_tokens=MAX_INPUT_TOKENS,
    generation_timeout_seconds=GENERATION_TIMEOUT_SECONDS,
    max_upload_mb=MAX_UPLOAD_MB,
=======
    max_new_tokens=MAX_NEW_TOKENS,
    max_input_tokens=MAX_INPUT_TOKENS,
    generation_timeout_seconds=GENERATION_TIMEOUT_SECONDS,
    max_upload_mb=MAX_UPLOAD_MB,
    max_pages=MAX_PAGES,
>>>>>>> REPLACE

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && grep -n 'chunk\|retriev\|Retriev\|BM25\|documents\|stored\|DELETE\|TTL' build_notebook.py"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && sed -i '197s/.*/print(\"  limits  :\", settings.max_input_tokens, \"prompt tokens,\", settings.max_new_tokens, \"new tokens,\", settings.max_pages, \"pages\")/' build_notebook.py && sed -n 195,199p build_notebook.py && sed -n 200,222p build_notebook.py"

### 🤖 Assistant

> 🔧 called: apply_diff, read_file, read_file



> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:197
-------
print("  chunks  :", settings.chunk_size, "chars, overlap", settings.chunk_overlap)
=======
print("  limits  :", settings.max_input_tokens, "prompt tokens,", settings.max_new_tokens, "new tokens,", settings.max_pages, "pages")
>>>>>>> REPLACE

> 🔧 **read_file** — colab/build_notebook.py

> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: apply_diff, execute_command

Two small text fixes in the Cell 2 and Cell 3 descriptions, plus the Cell 8b wording.

> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:118
-------
| `ADAPTER_PATH` | Folder containing the saved LoRA adapter (it holds `adapter_config.json`). Leave `""` to run the base model alone. |
=======
| `ADAPTER_PATH` | Folder containing the saved LoRA adapter (it holds `adapter_config.json`), i.e. the folder the training notebook wrote with `model.save_pretrained(f"{OUT}/adapter")`. If left `""`, the **un-fine-tuned** base model runs; it was never trained on the `### Answer:` format. |
>>>>>>> REPLACE
<<<<<<< SEARCH
:start_line:210
-------
* Model loading is deliberately separate from the upload/extraction/question functions below.
* It never substitutes another model: a missing or wrong path stops with an error.

This class also contains the generation method (`generate_answer`) used by Cell 8 and the API.
Loading a few-billion-parameter model can take several minutes.
=======
* Model loading is deliberately separate from the PDF extraction code below.
* It never substitutes another model: a missing or wrong path stops with an error.
* Like the training notebook, it sets `pad_token = eos_token` when the tokenizer has no pad token.

This class also contains the generation method (`generate_answer`) used by Cell 6 and the API.
>>>>>>> REPLACE

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && grep -n 'Documents held in memory' build_notebook.py"

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: apply_diff



> 🔧 **apply_diff** — <<<<<<< SEARCH
:start_line:426
-------
Run this when you are finished. It stops the public URL and the API. Documents held in memory are
discarded when the session ends. To start again, re-run Cell 8.
=======
Run this when you are finished. It stops the public URL and the API. (The server never stores
uploaded documents, so there is nothing to clean up.) To start again, re-run Cell 8.
>>>>>>> REPLACE

### 🤖 Assistant

> 🔧 called: write_file, execute_command

Now `verify_notebook.py` for the new cell layout and the single endpoint.

> 🔧 **write_file** — colab/verify_notebook.py

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/colab && ../slm-document-analysis-backend/.venv/bin/python build_notebook.py && cd ~/ARC/Project/colab && MODEL_NAME_OR_PATH=Qwen/Qwen1.5-0.5B timeout 250 ../slm-document-analysis-backend/.venv/bin/python verify_notebook.py 2>&1 | grep -v 'HTTP Request\|Loading weights\|Warning' | tail -40"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/frontend/src && wc -l *.ts *.tsx components/*.tsx styles.css && cat types.ts api.ts App.tsx config.ts utils.ts clipboard.ts"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -c "cd ~/ARC/Project/frontend/src && cat components/Header.tsx components/UploadCard.tsx components/QuestionCard.tsx components/AnswerCard.tsx components/TextPreviewCard.tsx styles.css api.test.ts"

### 🤖 Assistant

> 🔧 called: write_file, write_file, write_file

The backend and the Colab notebook are done. The single-submit frontend is next, starting with the types and the API client.

> 🔧 **write_file** — frontend/src/types.ts

> 🔧 **write_file** — frontend/src/api.ts

> 🔧 **write_file** — frontend/src/api.test.ts

### 🤖 Assistant

> 🔧 called: write_file, write_file, write_file, write_file, write_file

Now the components for the single-submit flow: Step 1 chooses the PDF, Step 2 takes the question and submits, Step 3 shows the answer, and Step 4 shows the structured text that was sent to the model.

> 🔧 **write_file** — import { useRef, useState } from "react";
import { formatBytes, validatePdfFile } from "../utils.ts";

interface Props {
  maxMb: number;
  file: File | null;
  disabled: boolean;
  onFileChange: (file: File | null) => void;
}

/** Step 1: choose a PDF. Nothing is uploaded until the question is submitted in Step 2. */
export function UploadCard({ maxMb, file, disabled, onFileChange }: Props) {
  const [fileError, setFileError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const choose = (candidate: File | undefined) => {
    if (!candidate) return;
    const problem = validatePdfFile(candidate, maxMb);
    if (problem) {
      onFileChange(null);
      setFileError(problem);
      return;
    }
    onFileChange(candidate);
    setFileError(null);
  };

  return (
    <section className="card" aria-labelledby="step1">
      <h2 id="step1">Step 1: Choose Your PDF</h2>

      <div className="notice notice-warn">
        <strong>Text-based PDFs only.</strong> This version reads the text already embedded in a PDF.
        Scanned or image-only PDFs have no extractable text and will be rejected; OCR is not implemented.
      </div>

      <div
        className={`dropzone${dragging ? " dropzone-active" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) choose(e.dataTransfer.files[0]);
        }}
      >
        <div className="dropzone-icon" aria-hidden="true">&#128196;</div>
        <p><strong>Drag and drop a PDF here</strong></p>
        <p className="muted small">or</p>
        <button type="button" className="btn btn-secondary" onClick={() => inputRef.current?.click()} disabled={disabled}>
          Browse Files
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(e) => {
            choose(e.target.files?.[0]);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
        <p className="muted small">PDF only, up to {maxMb} MB</p>
      </div>

      {file && (
        <p className="file-info">
          <span aria-hidden="true">&#128206;</span> <strong>{file.name}</strong>
          <span className="muted"> &middot; {formatBytes(file.size)}</span>
        </p>
      )}
      {fileError && <p className="error-text" role="alert">{fileError}</p>}
    </section>
  );
}


> 🔧 **write_file** — import { MAX_QUESTION_LENGTH } from "../config.ts";

// Suggestions only: clicking one fills the input. No answers are attached to them.
const EXAMPLE_QUESTIONS = [
  "What is the main topic of this document?",
  "What are the key conclusions or findings?",
  "Who is mentioned in the document?",
];

interface Props {
  question: string;
  onQuestionChange: (value: string) => void;
  /** Why submitting is currently impossible (null when it is possible). */
  blockedReason: string | null;
  asking: boolean;
  error: string | null;
  onSubmit: () => void;
}

/** Step 2: type the question and send the PDF + question to the backend in ONE request. */
export function QuestionCard({ question, onQuestionChange, blockedReason, asking, error, onSubmit }: Props) {
  const canSubmit = blockedReason === null && !asking;

  return (
    <section className="card" aria-labelledby="step2">
      <h2 id="step2">Step 2: Ask a Question</h2>

      <label htmlFor="question" className="visually-hidden">Your question</label>
      <textarea
        id="question"
        rows={4}
        value={question}
        maxLength={MAX_QUESTION_LENGTH}
        placeholder="Ask a question about your PDF..."
        disabled={asking}
        onChange={(e) => onQuestionChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSubmit) onSubmit();
        }}
      />

      <div className="chips" aria-label="Example questions">
        <span className="muted small">Try:</span>
        {EXAMPLE_QUESTIONS.map((q) => (
          <button key={q} type="button" className="chip" disabled={asking} onClick={() => onQuestionChange(q)}>
            {q}
          </button>
        ))}
      </div>

      <div className="button-row">
        <button type="button" className="btn btn-primary" disabled={!canSubmit} onClick={onSubmit}>
          {asking ? "Working..." : "Get Answer"}
        </button>
        {blockedReason && !asking && <span className="muted small">{blockedReason}</span>}
        {canSubmit && <span className="muted small">Tip: Cmd/Ctrl + Enter to submit</span>}
      </div>
      <p className="muted small">
        The PDF and the question are sent together in one request. The server keeps nothing afterwards,
        so each question sends the PDF again.
      </p>

      {asking && (
        <div className="status-row" role="status">
          <span className="spinner" aria-hidden="true" />
          Reading the PDF structure and generating an answer. This can take a while on a small GPU or CPU...
        </div>
      )}
      {error && <div className="notice notice-error" role="alert"><strong>Could not get an answer.</strong> {error}</div>}
    </section>
  );
}


> 🔧 **write_file** — import { useState } from "react";
import type { AnswerLocation, AnswerResponse } from "../types.ts";
import { copyToClipboard } from "../clipboard.ts";
import { formatNumber } from "../utils.ts";

interface Props {
  result: AnswerResponse | null;
  asking: boolean;
}

const ELEMENT_LABELS: Record<string, string> = {
  paragraph: "paragraph",
  list_item: "list item",
  table: "table",
  title: "title",
  heading_1: "heading",
  heading_2: "heading",
  heading_3: "heading",
};

/** Step 3: the model's answer, run statistics and where the answer text appears in the PDF. */
export function AnswerCard({ result, asking }: Props) {
  const [copied, setCopied] = useState<string | null>(null);

  if (!result) {
    return (
      <section className="card" aria-labelledby="step3">
        <h2 id="step3">Step 3: Generated Answer</h2>
        <p className="muted">Waiting for the model...</p>
      </section>
    );
  }

  const copy = async () => {
    const ok = await copyToClipboard(result.answer);
    setCopied(ok ? "Copied to clipboard." : "Copy failed.");
    window.setTimeout(() => setCopied(null), 3000);
  };

  return (
    <section className={`card${asking ? " card-stale" : ""}`} aria-labelledby="step3">
      <h2 id="step3">Step 3: Generated Answer</h2>
      {asking && <p className="muted small">Showing the previous answer while a new one is generated...</p>}

      <div className="qa-block">
        <div className="label">Question</div>
        <p>{result.question}</p>
      </div>

      <div className="qa-block">
        <div className="label">Answer</div>
        <p className="answer">{result.answer}</p>
      </div>

      {result.unanswerable && (
        <div className="notice notice-warn" role="status">
          {result.message ?? "The model answered 'unanswerable': it judged that the document does not contain the answer."}
        </div>
      )}
      {result.document_truncated && (
        <div className="notice notice-warn" role="alert">
          <strong>Only part of the document was used.</strong> The PDF was longer than the model's input
          limit, so only the first {formatNumber(result.document_chars_used)} of{" "}
          {formatNumber(result.document_chars_total)} characters were given to the model. The answer may
          miss information from the end of the document.
        </div>
      )}

      <dl className="meta">
        <div><dt>Model</dt><dd>{result.model_name}</dd></div>
        <div><dt>Inference time</dt><dd>{result.inference_time_seconds.toFixed(2)} s</dd></div>
        <div><dt>Prompt tokens</dt><dd>{formatNumber(result.prompt_tokens)}</dd></div>
        <div><dt>Generated tokens</dt><dd>{formatNumber(result.generated_tokens)}</dd></div>
      </dl>

      <div className="button-row">
        <button type="button" className="btn btn-secondary" onClick={copy}>Copy Answer</button>
        {copied && <span className="muted small" role="status">{copied}</span>}
      </div>

      {result.disclaimer && <p className="muted small">{result.disclaimer}</p>}

      {!result.unanswerable && (
        <div className="evidence">
          <h3>Where this answer text appears in the document</h3>
          {result.answer_locations.length > 0 ? (
            <>
              <p className="muted small">
                Found by searching the extracted document for the answer text. This is a text match,
                not an explanation from the model, and it does not prove the answer is correct.
              </p>
              <ol className="passages">
                {result.answer_locations.map((loc, i) => (
                  <li key={i} className="passage">
                    <LocationHead loc={loc} />
                    <blockquote>{loc.snippet}</blockquote>
                  </li>
                ))}
              </ol>
            </>
          ) : (
            <p className="muted small">
              The answer text was not found word-for-word in the document (the model may have
              rephrased, combined or computed it, or it may be wrong), so no location is shown.
            </p>
          )}
        </div>
      )}
    </section>
  );
}

function LocationHead({ loc }: { loc: AnswerLocation }) {
  const kind = ELEMENT_LABELS[loc.element_type] ?? loc.element_type;
  return (
    <div className="passage-head small muted">
      Page {loc.page_number} &middot; {kind}
      {loc.section_path.length > 0 && <> &middot; Section: {loc.section_path.join(" > ")}</>}
      {loc.table_cell && (
        <>
          {" "}
          &middot; Table cell
          {loc.table_cell.row_label ? <> &middot; row &ldquo;{loc.table_cell.row_label}&rdquo;</> : null}
          {loc.table_cell.column_label ? <> &middot; column &ldquo;{loc.table_cell.column_label}&rdquo;</> : null}
        </>
      )}
    </div>
  );
}


> 🔧 **write_file** — import { useState } from "react";
import type { AnswerResponse } from "../types.ts";
import { copyToClipboard } from "../clipboard.ts";
import { formatNumber } from "../utils.ts";

interface Props {
  result: AnswerResponse;
}

/** Step 4: the structure-preserving Markdown that the backend gave to the model. */
export function StructureCard({ result }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const { document: doc } = result;

  const copy = async () => {
    const ok = await copyToClipboard(result.structured_text);
    setCopied(ok ? "Copied to clipboard." : "Copy failed. Select the text and copy it manually.");
    window.setTimeout(() => setCopied(null), 3000);
  };

  return (
    <section className="card" aria-labelledby="step4">
      <h2 id="step4">Step 4: Document Structure Given to the Model</h2>
      <p className="muted small">
        This is the text extracted from <em>{doc.filename}</em> with headings (<code>#</code>), lists
        (<code>-</code>) and tables (<code>| col |</code>) preserved. The whole text is placed in the
        prompt; nothing is split into chunks or searched first.
      </p>

      <dl className="stats">
        <div><dt>Pages</dt><dd>{formatNumber(doc.page_count)}</dd></div>
        <div><dt>Headings</dt><dd>{formatNumber(doc.heading_count)}</dd></div>
        <div><dt>Tables</dt><dd>{formatNumber(doc.table_count)}</dd></div>
        <div><dt>Words</dt><dd>{formatNumber(doc.word_count)}</dd></div>
        <div><dt>Characters</dt><dd>{formatNumber(doc.character_count)}</dd></div>
      </dl>
      {doc.empty_page_numbers.length > 0 && (
        <p className="muted small">No text found on page(s): {doc.empty_page_numbers.join(", ")}.</p>
      )}

      <pre className={`text-box${expanded ? " text-box-expanded" : ""}`} tabIndex={0}>{result.structured_text}</pre>
      {result.structured_text_truncated && (
        <p className="muted small">
          Showing the first {formatNumber(result.structured_text.length)} characters only.
        </p>
      )}

      <div className="button-row">
        <button type="button" className="btn btn-secondary" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Collapse" : "Expand"}
        </button>
        <button type="button" className="btn btn-secondary" onClick={copy}>Copy Text</button>
        {copied && <span className="muted small" role="status">{copied}</span>}
      </div>
    </section>
  );
}


> 🔧 **write_file** — import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiRequestError, createApi } from "./api.ts";
import { API_BASE_URL, MAX_QUESTION_LENGTH, MAX_UPLOAD_MB } from "./config.ts";
import { AnswerCard } from "./components/AnswerCard.tsx";
import { Header } from "./components/Header.tsx";
import { QuestionCard } from "./components/QuestionCard.tsx";
import { StructureCard } from "./components/StructureCard.tsx";
import { UploadCard } from "./components/UploadCard.tsx";
import type { AnswerResponse, HealthResponse } from "./types.ts";

export type BackendState =
  | { kind: "checking" }
  | { kind: "online"; health: HealthResponse }
  | { kind: "offline"; message: string };

export default function App() {
  const api = useMemo(() => createApi(API_BASE_URL), []);

  const [backend, setBackend] = useState<BackendState>({ kind: "checking" });
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [resetNotice, setResetNotice] = useState<string | null>(null);
  const requestToken = useRef(0);

  const checkBackend = useCallback(async () => {
    setBackend({ kind: "checking" });
    try {
      setBackend({ kind: "online", health: await api.health() });
    } catch (err) {
      setBackend({ kind: "offline", message: errorMessage(err) });
    }
  }, [api]);

  useEffect(() => {
    void checkBackend();
  }, [checkBackend]);

  const trimmed = question.trim();
  const blockedReason = submitBlockedReason(file !== null, trimmed, backend);

  const handleSubmit = async () => {
    if (!file || blockedReason) return;
    const token = ++requestToken.current;
    setAsking(true);
    setError(null);
    setResetNotice(null);
    try {
      const response = await api.askAboutDocument(file, trimmed);
      if (token === requestToken.current) setResult(response);
    } catch (err) {
      if (token === requestToken.current) setError(errorMessage(err));
    } finally {
      if (token === requestToken.current) setAsking(false);
    }
  };

  // Nothing is stored on the server, so resetting only clears this page.
  const handleReset = () => {
    requestToken.current++; // ignore any in-flight answer
    setAsking(false);
    setFile(null);
    setQuestion("");
    setResult(null);
    setError(null);
    setResetNotice("Cleared. The server keeps no copy of your document, so nothing else needs deleting.");
  };

  return (
    <div className="page">
      <Header />
      <main className="container">
        <BackendBanner backend={backend} onRetry={checkBackend} />

        <UploadCard maxMb={MAX_UPLOAD_MB} file={file} disabled={asking} onFileChange={setFile} />

        <QuestionCard
          question={question}
          onQuestionChange={setQuestion}
          blockedReason={blockedReason}
          asking={asking}
          error={error}
          onSubmit={handleSubmit}
        />

        {(result || asking) && <AnswerCard result={result} asking={asking} />}
        {result && <StructureCard result={result} />}

        <section className="reset-row">
          <button type="button" className="btn btn-secondary" onClick={handleReset}>
            Start New Document / Reset
          </button>
          {resetNotice && <p className="muted small" role="status">{resetNotice}</p>}
        </section>

        <footer className="footer">
          Answers are generated by a small language model and may be incorrect. Your PDF is only read
          to answer the question; it does not train or fine-tune the model and is not stored by the server.
        </footer>
      </main>
    </div>
  );
}

function errorMessage(err: unknown): string {
  return err instanceof ApiRequestError ? err.message : "An unexpected error occurred.";
}

/** Why the "Get Answer" button is disabled, or null when a request may be sent. */
export function submitBlockedReason(hasFile: boolean, trimmedQuestion: string, backend: BackendState): string | null {
  if (backend.kind === "offline") return "The backend is unreachable.";
  if (backend.kind === "checking") return "Checking the backend...";
  if (!backend.health.model_ready) {
    return backend.health.model_error
      ? `The model is not ready: ${backend.health.model_error}`
      : "The model is not ready yet.";
  }
  if (!hasFile) return "Choose a PDF in Step 1.";
  if (trimmedQuestion.length === 0) return "Type a question.";
  if (trimmedQuestion.length > MAX_QUESTION_LENGTH) return `The question is longer than ${MAX_QUESTION_LENGTH} characters.`;
  return null;
}

function BackendBanner({ backend, onRetry }: { backend: BackendState; onRetry: () => void }) {
  if (backend.kind === "checking") {
    return <div className="banner banner-info" role="status">Checking backend connection...</div>;
  }
  if (backend.kind === "offline") {
    return (
      <div className="banner banner-error" role="alert">
        <div>
          <strong>Backend unavailable.</strong> {backend.message}
          <div className="small">Backend URL: <code>{API_BASE_URL}</code></div>
        </div>
        <button type="button" className="btn btn-small" onClick={onRetry}>Retry</button>
      </div>
    );
  }
  const { health } = backend;
  if (!health.model_ready) {
    return (
      <div className="banner banner-warn" role="alert">
        <div>
          <strong>Backend connected, but the model is not ready.</strong>{" "}
          {health.model_error ?? "Wait for model loading to finish in the Colab notebook."}
          <div className="small">Questions are disabled until the model is ready.</div>
        </div>
        <button type="button" className="btn btn-small" onClick={onRetry}>Re-check</button>
      </div>
    );
  }
  return (
    <div className="banner banner-ok" role="status">
      Backend connected. Model ready: <strong>{health.model_name ?? "unknown"}</strong>
      {health.device ? ` (device: ${health.device})` : ""}
    </div>
  );
}


### 🤖 Assistant

Previous conversation summary:

## Goal
The user is building an M.Tech proof of concept, **"A Study of Parameter-Efficient Fine-Tuning of Small Language Models (SLMs) for Document Analysis."**
- Frontend: React + TypeScript + Vite, run locally (WSL, `http://localhost:5173`).
- Backend: FastAPI, run in Google Colab and exposed through a cloudflared tunnel, with an optional local run in WSL.
- Model: the user's Qwen1.5-0.5B base plus a LoRA adapter, used for inference only.

The user's latest, major architecture request:
- Drop the "upload, then ask" flow and drop RAG, chunking, retrieval and server storage.
- The frontend sends the PDF and the typed question in **ONE request**.
- The backend extracts the text **preserving structure** (headings, lists, tables → Markdown).
- It builds **one unified prompt in the exact training format**, runs Qwen-0.5B + LoRA, and returns the answer.

## Instructions
These are strict and still apply.

**Prompt format**
- The format is exactly: `"### Context:\n{doc}\n\n### Question:\n{q}\n\n### Answer:\n"`.
- The training notebook is `notebooks/simple_pipeline.ipynb`, Step 3.
- There is no instruction sentence and no chat template.
- Decoding is greedy.
- Only newly generated tokens are decoded, and the **first line** is the answer.
- `unanswerable` is the training label.
- The base model is a base model, not an instruct model.
- `tok.pad_token = eos_token` if there is no pad token.
- Training used `MAX_LEN=384`.

**Structure formatting**
- Tables become Markdown tables (`| col |` with `---` separators).
- Headings use `#` / `##` markers.

**Provenance ("wow factor")**
- `locate_answer` string-matches the answer against the extracted elements. It reports page, section path, and table row and column labels, and only on a real match.
- It must **not** be presented as model output.
- Do not change the prompt to ask for reasoning.

**Reference folder**
- `notebooks/` is reference only and the user will delete it. Do not modify it, and do not depend on it.

**Earlier rules that still apply**
- Never fabricate answers, metrics, model names, evidence or confidence scores.
- Never claim OCR.
- Describe storage honestly: the server is now stateless.
- No secrets.
- Bind to 127.0.0.1.
- Follow the project's `security.md`:
  - no wildcard CORS
  - generic errors with no stack traces
  - no hardcoded secrets
- State exactly what was and wasn't tested.
- Never invent model paths. `MODEL_NAME_OR_PATH` and `ADAPTER_PATH` stay user-supplied. Qwen/Qwen1.5-0.5B may be mentioned as the likely base, but is not hardcoded.
- The frontend header has no "Proof of Concept" badge. Keep it that way.
- Keep the on-screen Step 1 → Step N UI style.

## Discoveries
**Environment**
- The project root is `\\wsl.localhost\Ubuntu\home\ardraj\ARC\Project` (WSL: `~/ARC/Project`).
- The user is on Windows with WSL Ubuntu. Node v22 is in WSL, loaded via `bash -lc` for nvm.
- The shell is PowerShell, and complex quoting of `wsl.exe -d Ubuntu -- bash -c "..."` breaks easily.
  - `sed -i` with quotes or apostrophes fails.
  - Heredocs containing backticks or `**` fail.
  - Use `apply_diff`, `write_file` or `search_and_replace` for edits instead.
- File tools are blocked outside the workspace.
- `.gitignore` cannot be written with `write_file`; use a shell command.
- **The `write_file` tool intermittently errors with "path parameter is required".** The `path` parameter must be given as an actual parameter, in the same call. When several write_file calls are issued in a single block, they often fail; issue them one at a time, or at most 2–3 at once, and verify.
- The `update_todo_list` tool rejects any list that drops earlier completed items, which includes the exact original text of "Inspect environment (no access to Mtech notebook/model config; Node via pip wheel in WSL)". Skip it, or reproduce all completed items verbatim.

**Python environment**
- The venv at `slm-document-analysis-backend/.venv` runs Python 3.14.
- Installed: fastapi, uvicorn, pymupdf 2.0, pydantic-settings, pytest, httpx2, torch CPU, transformers 5.22, peft 0.21, accelerate.
- transformers 5.x uses `dtype=`.
- `tests/` import with `PYTHONPATH=.`; pytest.ini sets `pythonpath = .`.
- HF downloads work from WSL.

**Verified results**
- The extractor on the generated structured PDF gives `title`, `heading_1`, `paragraph`, `heading_2`, 2 `list_item`s, a `heading_1`, a `table`, and a `paragraph`.
  - The Symbol bullet `·` needed adding to `BULLET_MARKER`.
- The real `Qwen/Qwen1.5-0.5B` base (no adapter) was run on CPU through `TestClient`, and `POST /answer` returned 200.
  - The prompt was 122 tokens and it generated 16 tokens in about 3.8 s.
  - The answer was "Profit of the North region was 15." That is wrong and expected, since there was no adapter. It shows the pipeline works and says nothing about quality.
- The old two-step design was fully built and tested earlier.

## Accomplished

### Backend (DONE, tested)
Folder: `slm-document-analysis-backend/`. **59 pytest tests pass and 3 are skipped.** The skipped ones are `requires_model`, and they were also run separately with the tiny model: 3 passed.

- New `app/structure.py`:
  - `DocumentElement`, `StructuredDocument`, `table_to_markdown`, `elements_to_markdown`, `locate_answer`.
  - `BULLET_MARKER`, `NUMBERED_MARKER`, `LIST_MARKER`.
  - The dataclass was renamed to `AnswerMatch`, because the pydantic `AnswerLocation` lives in `schemas.py`. The rename was done by sed, and `main.py` accesses its fields by attribute, so it works.
- New `app/document_processor.py`:
  - Structure extraction using PyMuPDF `get_text("dict")`.
  - Heading detection by font ratio (title ≥1.5, h1 ≥1.25, h2 ≥1.10, bold → h3, section numbers).
  - Bullets and numbered items → list items.
  - Ruled tables from `page.find_tables()`, merged by y position.
  - Header/footer and page-number removal.
  - Paragraph joining with de-hyphenation.
  - Validation errors, including a new `TooManyPagesError`.
  - `process_pdf(filename, data, *, max_bytes, min_text_chars, max_pages)` returns a `StructuredDocument`.
- New `app/prompting.py`: `build_prompt`, `prompt_parts(question)`, `first_line`, `is_unanswerable`.
- `app/model_service.py`:
  - `GenerationResult` has `answer`, `unanswerable`, `document_truncated`, `document_chars_total`, `document_chars_used`, plus the model and device fields.
  - `generate_answer(document_markdown, question)`.
  - A token budget is applied by binary-searching a cut at the END of the document only.
  - Greedy decoding.
- `app/config.py`:
  - Removed the chunk, retrieval and storage settings.
  - Defaults are `max_new_tokens=64`, `max_input_tokens=8192`, `max_pages=100`, `max_question_chars=2000`, and `max_markdown_response_chars=200000`.
- `app/schemas.py`: `HealthResponse`, `DocumentInfo`, `TableCell`, `AnswerLocation`, `AnswerResponse`.
- `app/main.py`:
  - `GET /health` and `POST /answer` (multipart `file` + `question`).
  - The model readiness check (503) runs before extraction.
  - Error codes: `empty_question`, `question_too_long`, and the upload/processing errors.
  - Status codes: 413 for `too_many_pages`, 504 for inference timeout.
  - CORS allows GET, POST and OPTIONS.
- Deleted: `chunking.py`, `retrieval.py`, `storage.py`, and the old tests.
- Tests written: `conftest.py` (with `make_pdf`, `make_structured_pdf` and `FakeModelService`), `test_document_processor.py`, `test_structure.py`, `test_api.py`, `test_model_integration.py`.
- Rewritten: `.env.example`, `scripts/smoke_test.sh`, backend `README.md`.
- `requirements.txt` is unchanged.

### Colab (DONE, verified)
Folder: `colab/`.
- `build_notebook.py` was restructured to **9 code cells** (cells 1–8 plus 8b):
  - 1 pip
  - 2 config
  - 3 model_service plus load
  - 4 document_processor
  - 5 structure plus prompting
  - 6 smoke test
  - 7 errors, schemas, main (create_app) plus the CORS check
  - 8 uvicorn plus the cloudflared tunnel
  - 8b stop
- The intro and descriptions were updated.
- `verify_notebook.py` was rewritten to exercise `/answer`.
- The notebook was rebuilt (19 cells), and `verify_notebook.py` was run locally with the real `Qwen/Qwen1.5-0.5B` on CPU. **All checks passed**:
  - health
  - single request
  - invalid PDF
  - blank question
  - missing question
  - old `/documents` gone
  - CORS preflight on `/answer`
- Not re-verified: the cloudflared tunnel, because this run used `START_TUNNEL=False`. That code is unchanged from the earlier version, which was tested successfully.

### Frontend (IN PROGRESS — partially done)
Folder: `frontend/` (`package.json`, `vite.config.ts`, etc.).

**Written successfully (and diffs shown):**
- `src/types.ts`: `HealthResponse`, `DocumentInfo`, `TableCell`, `AnswerLocation`, `AnswerResponse`.
- `src/api.ts`: `createApi` has `health()` and `askAboutDocument(file, question)` → `POST /answer` multipart. Answer timeout is 150 s.
- `src/api.test.ts`: rewritten for the new client.

**Attempted but FAILED** (all "path parameter is required"; the files are **still the OLD versions**), and they must be re-written, ideally one `write_file` per call with the `path` set:
- `frontend/src/components/UploadCard.tsx`:
  - Step 1.
  - Props: `maxMb`, `file`, `disabled`, `onFileChange`.
  - File choose / drag-drop, no upload button.
- `frontend/src/components/QuestionCard.tsx`:
  - Step 2.
  - Props: `question`, `onQuestionChange`, `blockedReason`, `asking`, `error`, `onSubmit`.
  - "Get Answer" button, example chips, a spinner.
  - Cmd/Ctrl+Enter submits.
  - Explains that the PDF is re-sent on each question.
- `frontend/src/components/AnswerCard.tsx`:
  - Step 3.
  - Shows question, answer, an "unanswerable" notice, and a truncation warning.
  - A meta list: model, inference time, prompt tokens, generated tokens.
  - Copy Answer, the disclaimer.
  - An evidence section, "Where this answer text appears in the document", from `answer_locations`.
    - Each location shows page, element type, section path, and table row and column labels, plus the snippet.
    - It says this is a text match, not model reasoning.
    - If there are no locations, it says "not found word-for-word".
- New `frontend/src/components/StructureCard.tsx`:
  - Step 4.
  - Shows the Markdown that was given to the model.
  - Stats for pages, headings, tables, words and characters.
  - Expand/collapse, Copy Text, and a note when `structured_text_truncated`.
- `frontend/src/App.tsx`:
  - State: `backend`, `file`, `question`, `asking`, `error`, `result`, `resetNotice`, `requestToken`.
  - Exports `submitBlockedReason(hasFile, trimmedQuestion, backend)`.
  - `handleSubmit` calls `api.askAboutDocument`.
  - `handleReset` clears the page only.
  - Renders `BackendBanner`, `UploadCard`, `QuestionCard`, `AnswerCard`, `StructureCard`, the reset row and the footer.
  - The footer says the PDF is not stored.

**Still to change in the frontend:**
- Delete `src/components/TextPreviewCard.tsx`, which references the removed API.
- Check `config.ts`, which has `MAX_QUESTION_LENGTH=2000` and its comment mentions "QuestionRequest"; optionally fix the comment.
- `Header.tsx`: optionally update the subtitle to the single-submit flow ("Upload a PDF, ask a question…"); keep it badge-free.
- `styles.css` is probably fine. It already has the `.stats`, `.meta`, `.evidence`, `.passages`, `.passage`, `.text-box` and `.notice` classes.
- Run `npm run build` (this includes `tsc`) and `npm test` in `frontend/` via `bash -lc` in WSL; vitest should now pass 7–8 api tests plus the `utils` tests.

### Still to do
1. Finish the frontend (above), then run build and test until they pass.
2. Optional: rebuild a Playwright e2e. The old tooling was deleted. Skipping it is acceptable; say so.
3. Rewrite the root `README.md` (`~/ARC/Project/README.md`). It still documents the old two-step flow, so cover:
   - the new single-request architecture
   - the project structure
   - Colab steps (cells 1–8)
   - the frontend `.env` with `VITE_API_BASE_URL`
   - the model/adapter placeholders
   - Qwen as the likely base
   - the `locate_answer` explanation
   - the 384-token training caveat
   - an honest tested / not-tested section
4. Check `frontend/.env.example`, which has `VITE_API_BASE_URL` and `VITE_MAX_UPLOAD_MB` (probably fine).
5. Cleanup:
   - Remove `frontend/dist` if present.
   - Remove temp files in `/tmp` (not in the repo), e.g. `try_extract.py`, `real_e2e.py`, `patch_nb.py`.
   - Optionally remove `.pytest_cache` and the `__pycache__` directories.
   - Do not touch `notebooks/`.
6. Final summary to the user.
   - Tested:
     - 59 backend tests
     - the tiny-model integration
     - the real Qwen1.5-0.5B base on CPU through the API and through the notebook-verification script
     - the frontend build and unit tests once done
   - NOT tested:
     - Colab itself
     - the cloudflared tunnel with the new notebook
     - the user's real adapter
     - GPU
     - answer quality, especially on multi-page documents versus the ≤384-token training
     - Chrome on Windows with the new UI
     - the Mac / MPS
   - Remaining limitations:
     - heuristic structure detection
     - no OCR
     - tables without ruling lines
     - the tail of the document is cut when it is over the token limit

## Relevant files / directories
**Root:** `~/ARC/Project/`
- `README.md` (OLD, needs a rewrite)
- `notebooks/` (user's reference, do not modify)
- `colab/`
  - `build_notebook.py`, `verify_notebook.py`, `slm_document_analysis_backend.ipynb` (regenerated), `requirements.txt`
- `slm-document-analysis-backend/`
  - `app/`: `main.py`, `config.py`, `errors.py`, `schemas.py`, `structure.py`, `document_processor.py`, `prompting.py`, `model_service.py`
  - `tests/`: `conftest.py`, `test_document_processor.py`, `test_structure.py`, `test_api.py`, `test_model_integration.py`
  - `scripts/smoke_test.sh`, `.env.example`, `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`, `.venv/`
- `frontend/`
  - `src/`
    - `api.ts`, `api.test.ts`, `types.ts` (NEW versions done)
    - `App.tsx` (new version NOT yet written)
    - `config.ts`, `utils.ts`, `utils.test.ts`, `clipboard.ts`, `styles.css`, `main.tsx`
  - `src/components/`
    - `Header.tsx`
    - `UploadCard.tsx`, `QuestionCard.tsx`, `AnswerCard.tsx` (OLD, must be rewritten)
    - `TextPreviewCard.tsx` (to delete)
    - `StructureCard.tsx` (to create)
  - `package.json`, `vite.config.ts`, `.env.example`, `node_modules`



---

### 👤 User

continue

---

### 🤖 Assistant

> 🔧 called: list_files, read_file



> 🔧 **list_files** — frontend/src

> 🔧 **read_file** — frontend/src/App.tsx

### 🤖 Assistant

> 🔧 called: read_file, read_file, read_file, read_file, read_file, read_file, read_file, read_file



> 🔧 **read_file** — frontend/src/types.ts

> 🔧 **read_file** — frontend/src/utils.ts

> 🔧 **read_file** — frontend/src/config.ts

> 🔧 **read_file** — frontend/src/clipboard.ts

> 🔧 **read_file** — frontend/src/components/QuestionCard.tsx

> 🔧 **read_file** — frontend/src/components/UploadCard.tsx

> 🔧 **read_file** — frontend/src/components/AnswerCard.tsx

> 🔧 **read_file** — frontend/src/components/TextPreviewCard.tsx

### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — frontend/src/components/UploadCard.tsx

### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — import { MAX_QUESTION_LENGTH } from "../config.ts";

// Suggestions only: clicking one fills the input. No answers are attached to them.
const EXAMPLE_QUESTIONS = [
  "What is the main topic of this document?",
  "What are the key conclusions or findings?",
  "Who is mentioned in the document?",
];

interface Props {
  question: string;
  onQuestionChange: (value: string) => void;
  /** Why submitting is currently impossible (no file, backend offline...), or null when allowed. */
  blockedReason: string | null;
  asking: boolean;
  error: string | null;
  onSubmit: () => void;
}

/** Step 2: type the question and submit. The PDF and the question go to the backend together. */
export function QuestionCard({ question, onQuestionChange, blockedReason, asking, error, onSubmit }: Props) {
  const canSubmit = blockedReason === null && !asking;

  return (
    <section className="card" aria-labelledby="step2">
      <h2 id="step2">Step 2: Ask a Question</h2>

      <label htmlFor="question" className="visually-hidden">Your question</label>
      <textarea
        id="question"
        rows={4}
        value={question}
        maxLength={MAX_QUESTION_LENGTH}
        placeholder="Ask a question about your chosen document..."
        disabled={asking}
        onChange={(e) => onQuestionChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canSubmit) onSubmit();
        }}
      />

      <div className="chips" aria-label="Example questions">
        <span className="muted small">Try:</span>
        {EXAMPLE_QUESTIONS.map((q) => (
          <button key={q} type="button" className="chip" disabled={asking} onClick={() => onQuestionChange(q)}>
            {q}
          </button>
        ))}
      </div>

      <div className="button-row">
        <button type="button" className="btn btn-primary" disabled={!canSubmit} onClick={onSubmit}>
          {asking ? "Generating..." : "Get Answer"}
        </button>
        {blockedReason && <span className="muted small">{blockedReason}</span>}
        {!blockedReason && <span className="muted small">Tip: Cmd/Ctrl + Enter to submit</span>}
      </div>
      <p className="muted small">
        The PDF is sent to the backend together with your question, and it is sent again for every
        new question. The backend does not keep it.
      </p>

      {asking && (
        <div className="status-row" role="status">
          <span className="spinner" aria-hidden="true" />
          Extracting the document structure and generating an answer. This can take a while on a small GPU or CPU...
        </div>
      )}
      {error && <div className="notice notice-error" role="alert"><strong>Could not get an answer.</strong> {error}</div>}
    </section>
  );
}


### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — import { useState } from "react";
import type { AnswerLocation, AnswerResponse } from "../types.ts";
import { copyToClipboard } from "../clipboard.ts";
import { formatNumber } from "../utils.ts";

interface Props {
  result: AnswerResponse | null;
  asking: boolean;
}

/** Step 3: the model's answer, plus where that answer text literally occurs in the document. */
export function AnswerCard({ result, asking }: Props) {
  const [copied, setCopied] = useState<string | null>(null);

  if (!result) {
    return (
      <section className="card" aria-labelledby="step3">
        <h2 id="step3">Step 3: Generated Answer</h2>
        <p className="muted">Waiting for the model...</p>
      </section>
    );
  }

  const copy = async () => {
    if (!result.answer) return;
    const ok = await copyToClipboard(result.answer);
    setCopied(ok ? "Copied to clipboard." : "Copy failed.");
    window.setTimeout(() => setCopied(null), 3000);
  };

  return (
    <section className={`card${asking ? " card-stale" : ""}`} aria-labelledby="step3">
      <h2 id="step3">Step 3: Generated Answer</h2>
      {asking && <p className="muted small">Showing the previous answer while a new one is generated...</p>}

      <div className="qa-block">
        <div className="label">Question</div>
        <p>{result.question}</p>
      </div>

      <div className="qa-block">
        <div className="label">Answer</div>
        <p className="answer">{result.answer || "(the model produced an empty answer)"}</p>
      </div>

      {result.unanswerable && (
        <div className="notice notice-warn" role="status">
          <strong>The model answered "unanswerable".</strong>{" "}
          {result.message ?? "It did not find an answer in the supplied document text."}
        </div>
      )}
      {result.document_truncated && (
        <div className="notice notice-warn" role="status">
          <strong>The document was too long for the model.</strong> Only the first{" "}
          {formatNumber(result.document_chars_used)} of {formatNumber(result.document_chars_total)}{" "}
          characters were given to the model; the end was cut off.
        </div>
      )}

      <dl className="meta">
        <div><dt>Model</dt><dd>{result.model_name}</dd></div>
        <div><dt>Inference time</dt><dd>{result.inference_time_seconds.toFixed(2)} s</dd></div>
        <div><dt>Prompt tokens</dt><dd>{formatNumber(result.prompt_tokens)}</dd></div>
        <div><dt>Generated tokens</dt><dd>{formatNumber(result.generated_tokens)}</dd></div>
      </dl>

      <div className="button-row">
        <button type="button" className="btn btn-secondary" onClick={copy} disabled={!result.answer}>
          Copy Answer
        </button>
        {copied && <span className="muted small" role="status">{copied}</span>}
      </div>

      {result.disclaimer && <p className="muted small">{result.disclaimer}</p>}

      <div className="evidence">
        <h3>Where this answer text appears in the document</h3>
        <p className="muted small">
          Found by plain text matching of the answer against the extracted document. This is not
          produced by the model and is not a confidence score or an explanation of its reasoning.
        </p>
        {result.answer_locations.length === 0 ? (
          <p className="muted">
            The answer text was not found word-for-word in the document, so no location is shown.
          </p>
        ) : (
          <ol className="passages">
            {result.answer_locations.map((loc, i) => (
              <li key={i} className="passage">
                <div className="passage-head small muted">{describeLocation(loc)}</div>
                <blockquote>{loc.snippet}</blockquote>
              </li>
            ))}
          </ol>
        )}
      </div>
    </section>
  );
}

function describeLocation(loc: AnswerLocation): string {
  const parts = [`Page ${loc.page_number}`, loc.element_type.replace(/_/g, " ")];
  if (loc.section_path.length > 0) parts.push(`Section: ${loc.section_path.join(" > ")}`);
  const cell = loc.table_cell;
  if (cell) {
    if (cell.row_label) parts.push(`Row: ${cell.row_label}`);
    if (cell.column_label) parts.push(`Column: ${cell.column_label}`);
  }
  return parts.join(" \u00b7 ");
}


### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — import { useState } from "react";
import type { AnswerResponse } from "../types.ts";
import { copyToClipboard } from "../clipboard.ts";
import { formatNumber } from "../utils.ts";

interface Props {
  result: AnswerResponse;
}

/** Step 4: the Markdown the model actually received, so the extraction can be inspected. */
export function StructureCard({ result }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const doc = result.document;

  const copy = async () => {
    const ok = await copyToClipboard(result.structured_text);
    setCopied(ok ? "Copied to clipboard." : "Copy failed. Select the text and copy it manually.");
    window.setTimeout(() => setCopied(null), 3000);
  };

  return (
    <section className="card" aria-labelledby="step4">
      <h2 id="step4">Step 4: Extracted Document Structure</h2>
      <p className="muted small">
        The text of <em>{doc.filename}</em> as converted to Markdown (headings, lists, tables) before it
        was placed in the model's prompt.
      </p>

      <dl className="stats">
        <div><dt>Pages</dt><dd>{formatNumber(doc.page_count)}</dd></div>
        <div><dt>Headings</dt><dd>{formatNumber(doc.heading_count)}</dd></div>
        <div><dt>Tables</dt><dd>{formatNumber(doc.table_count)}</dd></div>
        <div><dt>Words</dt><dd>{formatNumber(doc.word_count)}</dd></div>
        <div><dt>Characters</dt><dd>{formatNumber(doc.character_count)}</dd></div>
      </dl>

      {doc.empty_page_numbers.length > 0 && (
        <p className="muted small">No text found on page(s): {doc.empty_page_numbers.join(", ")}.</p>
      )}

      <pre className={`text-box${expanded ? " text-box-expanded" : ""}`} tabIndex={0}>{result.structured_text}</pre>
      {result.structured_text_truncated && (
        <p className="muted small">
          This display is shortened. The model still received the document text as described above.
        </p>
      )}

      <div className="button-row">
        <button type="button" className="btn btn-secondary" onClick={() => setExpanded((v) => !v)}>
          {expanded ? "Collapse" : "Expand"}
        </button>
        <button type="button" className="btn btn-secondary" onClick={copy}>
          Copy Text
        </button>
        {copied && <span className="muted small" role="status">{copied}</span>}
      </div>
    </section>
  );
}


### 🤖 Assistant

> 🔧 called: search_and_replace, write_file, execute_command



> 🔧 **search_and_replace** — frontend/src/components/StructureCard.tsx

> 🔧 **write_file** — import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiRequestError, createApi } from "./api.ts";
import { API_BASE_URL, MAX_QUESTION_LENGTH, MAX_UPLOAD_MB } from "./config.ts";
import { AnswerCard } from "./components/AnswerCard.tsx";
import { Header } from "./components/Header.tsx";
import { QuestionCard } from "./components/QuestionCard.tsx";
import { StructureCard } from "./components/StructureCard.tsx";
import { UploadCard } from "./components/UploadCard.tsx";
import type { AnswerResponse, HealthResponse } from "./types.ts";

export type BackendState =
  | { kind: "checking" }
  | { kind: "online"; health: HealthResponse }
  | { kind: "offline"; message: string };

/** Returns why a question cannot be submitted right now, or null when it can. */
export function submitBlockedReason(
  hasFile: boolean,
  trimmedQuestion: string,
  backend: BackendState,
): string | null {
  if (!hasFile) return "Choose a PDF in Step 1 first.";
  if (trimmedQuestion.length === 0) return "Type a question first.";
  if (trimmedQuestion.length > MAX_QUESTION_LENGTH) return `The question is longer than ${MAX_QUESTION_LENGTH} characters.`;
  if (backend.kind === "offline") return "The backend is unreachable.";
  if (backend.kind === "checking") return "Checking the backend...";
  if (!backend.health.model_ready) {
    return backend.health.model_error
      ? `The model is not ready: ${backend.health.model_error}`
      : "The model is not ready yet.";
  }
  return null;
}

export default function App() {
  const api = useMemo(() => createApi(API_BASE_URL), []);

  const [backend, setBackend] = useState<BackendState>({ kind: "checking" });
  const [file, setFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [resetNotice, setResetNotice] = useState<string | null>(null);
  // Bumped on reset so UploadCard drops its local validation message.
  const [resetCount, setResetCount] = useState(0);
  const requestToken = useRef(0);

  const checkBackend = useCallback(async () => {
    setBackend({ kind: "checking" });
    try {
      setBackend({ kind: "online", health: await api.health() });
    } catch (err) {
      setBackend({ kind: "offline", message: errorMessage(err) });
    }
  }, [api]);

  useEffect(() => {
    void checkBackend();
  }, [checkBackend]);

  const trimmed = question.trim();
  const blockedReason = submitBlockedReason(file !== null, trimmed, backend);

  const handleSubmit = async () => {
    if (!file || blockedReason !== null) return;
    const token = ++requestToken.current;
    setAsking(true);
    setError(null);
    setResetNotice(null);
    try {
      const response = await api.askAboutDocument(file, trimmed);
      if (token === requestToken.current) setResult(response);
    } catch (err) {
      if (token === requestToken.current) setError(errorMessage(err));
    } finally {
      if (token === requestToken.current) setAsking(false);
    }
  };

  // Clears this page only. The backend keeps nothing, so there is nothing to delete there.
  const handleReset = () => {
    requestToken.current++; // ignore any in-flight answer
    setAsking(false);
    setFile(null);
    setQuestion("");
    setResult(null);
    setError(null);
    setResetCount((n) => n + 1);
    setResetNotice("Cleared. The backend does not store documents, so nothing needs to be deleted there.");
  };

  return (
    <div className="page">
      <Header />
      <main className="container">
        <BackendBanner backend={backend} onRetry={checkBackend} />

        <UploadCard
          key={`upload-${resetCount}`}
          maxMb={MAX_UPLOAD_MB}
          file={file}
          disabled={asking}
          onFileChange={setFile}
        />

        <QuestionCard
          question={question}
          onQuestionChange={setQuestion}
          blockedReason={blockedReason}
          asking={asking}
          error={error}
          onSubmit={handleSubmit}
        />

        {(result || asking) && <AnswerCard result={result} asking={asking} />}
        {result && <StructureCard result={result} />}

        <section className="reset-row">
          <button type="button" className="btn btn-secondary" onClick={handleReset}>
            Start New Document / Reset
          </button>
          {resetNotice && <p className="muted small" role="status">{resetNotice}</p>}
        </section>

        <footer className="footer">
          Answers are generated by a small language model and may be incorrect. The PDF is processed
          for question answering only and is not stored by the backend; it does not train or fine-tune
          the model.
        </footer>
      </main>
    </div>
  );
}

function errorMessage(err: unknown): string {
  return err instanceof ApiRequestError ? err.message : "An unexpected error occurred.";
}

function BackendBanner({ backend, onRetry }: { backend: BackendState; onRetry: () => void }) {
  if (backend.kind === "checking") {
    return <div className="banner banner-info" role="status">Checking backend connection...</div>;
  }
  if (backend.kind === "offline") {
    return (
      <div className="banner banner-error" role="alert">
        <div>
          <strong>Backend unavailable.</strong> {backend.message}
          <div className="small">Backend URL: <code>{API_BASE_URL}</code></div>
        </div>
        <button type="button" className="btn btn-small" onClick={onRetry}>Retry</button>
      </div>
    );
  }
  const { health } = backend;
  if (!health.model_ready) {
    return (
      <div className="banner banner-warn" role="alert">
        <div>
          <strong>Backend connected, but the model is not ready.</strong>{" "}
          {health.model_error ?? "Wait for model loading to finish in the Colab notebook."}
          <div className="small">Questions are disabled until the model is ready.</div>
        </div>
        <button type="button" className="btn btn-small" onClick={onRetry}>Re-check</button>
      </div>
    );
  }
  return (
    <div className="banner banner-ok" role="status">
      Backend connected. Model ready: <strong>{health.model_name ?? "unknown"}</strong>
      {health.device ? ` (device: ${health.device})` : ""}
    </div>
  );
}


> 🔧 **execute_command** — Remove-Item frontend/src/components/TextPreviewCard.tsx; Get-Content frontend/src/components/Header.tsx; Select-String -Path frontend/src/*.ts*,frontend/src/components/*.tsx -Pattern "TextPreview|QuestionRequest|uploadDocument|askQuestion"

### 🤖 Assistant

> 🔧 called: search_and_replace, grep



> 🔧 **search_and_replace** — frontend/src/config.ts

> 🔧 **grep** — max_question_chars|MAX_QUESTION

### 🤖 Assistant

> 🔧 called: search_and_replace, execute_command



> 🔧 **search_and_replace** — frontend/src/components/Header.tsx

> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/frontend && npm run build 2>&1 | tail -25 && npm test 2>&1 | tail -25"

### 🤖 Assistant

> 🔧 called: write_file, write_file



> 🔧 **write_file** — frontend/src/app.test.ts

> 🔧 **write_file** — frontend/src/components/Header.tsx

### 🤖 Assistant

> 🔧 called: execute_command, read_file



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/frontend && npm run build 2>&1 | tail -8 && npm test 2>&1 | tail -15; rm -rf dist"

> 🔧 **read_file** — README.md

### 🤖 Assistant

> 🔧 called: read_file, grep



> 🔧 **read_file** — slm-document-analysis-backend/app/config.py

> 🔧 **grep** — ^# ?(Cell|CELL)|^# =+|EDIT THESE|^[A-Z_]+ *=

### 🤖 Assistant

> 🔧 called: read_file



> 🔧 **read_file** — colab/build_notebook.py

### 🤖 Assistant

> 🔧 called: grep, read_file



> 🔧 **read_file** — slm-document-analysis-backend/app/schemas.py

> 🔧 **grep** — slm-document-analysis-backend/app

### 🤖 Assistant

> 🔧 called: grep, read_file



> 🔧 **read_file** — colab/build_notebook.py

> 🔧 **grep** — slm-document-analysis-backend/app

### 🤖 Assistant

> 🔧 called: read_file, read_file



> 🔧 **read_file** — slm-document-analysis-backend/app/main.py

> 🔧 **read_file** — slm-document-analysis-backend/README.md

### 🤖 Assistant

> 🔧 called: write_file



> 🔧 **write_file** — README.md

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project && cat frontend/.env.example; rm -rf frontend/dist slm-document-analysis-backend/.pytest_cache; find . -name __pycache__ -not -path '*/node_modules/*' -not -path '*/.venv/*' -prune -exec rm -rf {} +; rm -f /tmp/try_extract.py /tmp/real_e2e.py /tmp/patch_nb.py; grep -rniE 'chunk|retriev|/documents|BM25|upload and process' --include=*.ts --include=*.tsx --include=*.md --include=*.css --include=*.example --include=*.sh . --exclude-dir=node_modules --exclude-dir=.venv --exclude-dir=notebooks | head -20; git status --short | head -30"

### 🤖 Assistant

> 🔧 called: execute_command



> 🔧 **execute_command** — wsl.exe -d Ubuntu -- bash -lc "cd ~/ARC/Project/slm-document-analysis-backend && .venv/bin/python -m pytest -q -p no:cacheprovider 2>&1 | tail -4; find . -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +"

### 🤖 Assistant

The single-request version is finished: the browser sends the PDF and the question in one request to `POST /answer`, and the backend returns the answer. The backend, the Colab notebook and the frontend are all updated, but I have not tried it with your adapter, in Colab, or in a browser.

## How it works
- The backend reads the PDF with its structure kept (headings, lists, ruled tables as Markdown).
- It builds the prompt in your training format: `### Context:\n…\n\n### Question:\n…\n\n### Answer:\n`.
- Qwen plus the LoRA adapter generates the answer with greedy decoding, and the first line is returned as the answer.
- Nothing is stored on the server and there is no chunking or retrieval.
- The frontend has four steps:
  1. Choose a PDF.
  2. Type a question and press **Get Answer**.
  3. See the answer with model name, timing, token counts, and where the answer text appears in the document.
  4. See the Markdown that the model received.
- "Where the answer text appears" is plain string matching of the answer against the extracted document. The UI says this is not model output.
- The PDF is sent again with every question.

## What I ran
- **Backend:** 59 pytest tests pass and 3 are skipped. The 3 skipped tests need a model, so I ran them separately with a tiny random model, and they passed.
- **Real Qwen1.5-0.5B base, no adapter, CPU:**
  - `POST /answer` returned 200 with a 122-token prompt and 16 new tokens in about 3.8 s.
  - The answer was wrong, which is expected without your adapter, so this only shows the pipeline runs.
  - The Colab notebook's cells also pass `colab/verify_notebook.py` with this model.
- **Frontend:** 17 unit tests pass, and the type-check and production build pass.

## Not tested
- Colab itself.
- The cloudflared tunnel with the new notebook. The same tunnel code worked in an earlier version.
- Your real adapter and GPU inference.
- Answer quality, especially on multi-page documents. Your adapter was trained on examples of at most 384 tokens, so training doesn't tell us how well it handles long documents.
- The new UI in a browser against the new backend. I did not rebuild the browser end-to-end check.
- macOS and Apple-Silicon.

## Limitations
- There is no OCR, so scanned PDFs are rejected.
- Structure detection is heuristic. Tables without ruling lines are read as plain text.
- If a document is too long for the model's input limit, the end is cut and the response reports it.
- There is no authentication, so keep the tunnel URL private.

## What you need to do
1. Open `colab/slm_document_analysis_backend.ipynb` in Colab. In Cell 2, set `MODEL_NAME_OR_PATH` and `ADAPTER_PATH`, and set `MOUNT_GOOGLE_DRIVE` if the adapter is in Drive. Run Cells 1–8.
2. Copy the `VITE_API_BASE_URL=…` line that Cell 8 prints into `frontend/.env`.
3. Run `npm run dev` in `frontend/` and open http://localhost:5173.

The root `README.md` is rewritten for this flow and covers setup, the API, the test procedure and the limitations. The full API reference and error codes are in `slm-document-analysis-backend/README.md`. I did not touch `notebooks/`.