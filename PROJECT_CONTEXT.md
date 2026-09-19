# PROJECT CONTEXT

## 1. Project Overview

This is a small, local Python prototype named **Explainable Semantic Plagiarism Detection** (the name used by the Flask API and Streamlit UI). It compares two uploaded documents at sentence level, calculates both lexical TF-IDF and semantic SBERT similarity, selects one source sentence as the best candidate for each submitted sentence, labels the candidate, produces a template explanation, and can persist the analysis in SQLite.

The application is split into a Flask HTTP API (`backend/app.py`), a currently independent Streamlit page (`frontend/app.py`), service modules, SQLAlchemy models, sample PDF inputs, saved JSON examples, and standalone test/demo scripts. It is not a completed end-user product: the frontend has no upload or API integration, and several operational concerns are deliberately absent.

## 2. Current Project Status

The core document-processing and similarity pipeline is implemented in code and saved result artifacts demonstrate a two-sentence PDF comparison. The REST API and database read/write services are implemented, but database table creation is not part of application startup. The local Streamlit page only displays a title and success message.

No Git repository is discoverable at this directory or its parent directories at inspection time. Therefore, the current branch, commit history, tracked/untracked state, staged changes, and a comparison with `HEAD` are **Unknown / Not determinable from the repository**. No `.git` directory was present. This also means there is no evidence that a given project file is a local modification versus its baseline version.

## 3. Technology Stack

| Area | Technology actually used in source |
| --- | --- |
| Language | Python |
| HTTP backend | Flask 3.1.3 |
| UI | Streamlit 1.62.0 |
| Database | SQLite through SQLAlchemy 2.0.52 |
| PDF reading | PyMuPDF (`pymupdf`) |
| DOCX reading | python-docx |
| Lexical similarity | scikit-learn `TfidfVectorizer` and cosine similarity |
| Semantic similarity | Sentence Transformers `all-MiniLM-L6-v2`, then scikit-learn cosine similarity |
| ML runtime | PyTorch / Transformers dependencies support Sentence Transformers |
| Serialization | Python `json` |

There is no source evidence of an external LLM API, embeddings database, authentication provider, cloud service, queue, web crawler, or deployment target. The SBERT model is a pretrained Hugging Face/Sentence Transformers model; model acquisition/cache behavior depends on the local Sentence Transformers environment and may require network access when it is not cached.

## 4. Repository Structure

```text
batch8-main/
├── backend/
│   ├── app.py                       Flask API entry point
│   ├── database/
│   │   ├── database.py              SQLite engine, session factory, declarative Base
│   │   └── models.py                Analysis and MatchResult ORM models
│   ├── services/                    document, ML, matching, explanation, and persistence services
│   ├── routes/                      empty package; no blueprint/routes live here
│   └── utils/                       empty package
├── frontend/
│   └── app.py                       minimal standalone Streamlit page
├── data/
│   ├── source.pdf                   sample source PDF (one page)
│   ├── submitted.pdf                sample submitted PDF (one page)
│   ├── test.pdf                     sample document-processing PDF (one page)
│   └── uploads/                     runtime upload destination; ignored by Git configuration
├── results/
│   ├── final_analysis.json          saved example pipeline output
│   └── similarity_comparison.json   saved all-pair similarity example
├── tests/                           executable demo/check scripts, not pytest assertion tests
├── requirements.txt                 dependency lock-style list
├── requirements-local.txt           alternate local dependency list
└── .gitignore                       ignores venv, bytecode, .env, SQLite DB, uploads
```

`.venv/` exists locally and is excluded from this map as a virtual environment. There are no README files, Dockerfiles, Compose files, CI/CD configuration, migration files, `.env` files, package manifests, or deployment configuration discovered.

## 5. Architecture

```text
Caller / Flask test client
        |
        | multipart POST /api/analyze
        v
Flask app (backend/app.py) ---- saves files ---> data/uploads/
        |
        v
plagiarism_pipeline.analyze_documents
        |
        +--> document_processor (PDF, DOCX, TXT -> page-tagged sentences)
        +--> preprocessing (adds TF-IDF-normalized text)
        +--> similarity (TF-IDF matrix + SBERT cosine matrix)
        +--> passage_matcher (best SBERT source candidate per submitted sentence)
        +--> plagiarism_detector (rule label)
        +--> explanation (percentages + static explanation text)
        |
        v
database_service ---> SQLAlchemy ---> plagiarism.db (SQLite, when initialized)
        |
        v
JSON HTTP response

Streamlit frontend (frontend/app.py)
        |
        +--> currently no connection to the Flask API
```

The API also reads history through `history_service`, which queries `Analysis` and its `MatchResult` relationship. There are no background jobs; analysis runs synchronously during the request.

## 6. Data Flow

1. `/api/analyze` requires multipart fields named `source_file` and `submitted_file`.
2. The API writes each upload under `data/uploads/<client filename>`.
3. `process_document` dispatches by suffix: `.pdf`, `.docx`, or `.txt`. It returns sentence dictionaries containing `sentence_id`, `page_number`, and original `text`.
4. `prepare_sentences` preserves `text` and adds lowercase, punctuation-stripped `tfidf_text`.
5. The pipeline calculates every source/submitted pair twice: a TF-IDF cosine matrix and an SBERT cosine matrix.
6. `find_best_matches` keeps the highest-SBERT candidate for each submitted sentence whose SBERT score meets `minimum_similarity`; the API supplies `0.0`.
7. The selected candidates get a score-threshold classification and a deterministic English explanation. The API stores them, then returns them.

For the existing sample artifacts, two submitted sentences yielded two matches: an exact sentence match and a near-paraphrase. The artifacts are examples, not proof of a currently runnable database/API session.

## 7. Entry Points

- `backend/app.py`: running `python backend/app.py` starts Flask at `127.0.0.1:5000` with `debug=True`. Importing this module creates `data/uploads/` if needed.
- `frontend/app.py`: intended for `streamlit run frontend/app.py`; it renders static text only.
- `tests/*.py`: individual ad hoc scripts intended to be run as Python files. Many carry out real file/database writes, so they are not read-only tests.

## 8. Core Modules

### Document and preprocessing

- `backend/services/document_processor.py`
  - `extract_pdf_text`, `extract_docx_text`, and `extract_txt_text` return page dictionaries. PDF page numbers are retained; DOCX and TXT are assigned page 1.
  - `extract_text` dispatches on lowercased file extension and raises `ValueError` for unsupported types.
  - `normalize_text` collapses whitespace; `split_sentences` uses the regex `(?<=[.!?])\s+`.
  - `process_document` produces sequential, one-based sentence IDs per document.
- `backend/services/preprocessing.py`
  - `normalize_for_tfidf` lowercases text, removes punctuation, and collapses whitespace.
  - `prepare_sentences` adds `tfidf_text` without changing the displayed original text.

### Similarity and decisioning

- `backend/services/tfidf_model.py`: `calculate_tfidf_similarity` fits a new `TfidfVectorizer` on the combined source and submitted sentence text, then returns source-by-submitted cosine similarity.
- `backend/services/sbert_model.py`: creates a module-global `SentenceTransformer("all-MiniLM-L6-v2")` at import time. `generate_embeddings` encodes original sentences into normalized vectors; `calculate_sbert_similarity` computes cosine similarity.
- `backend/services/similarity.py`: `calculate_all_similarities` combines the two matrices into a flat list of dictionaries with sentence IDs, page numbers, original text, and both scores.
- `backend/services/passage_matcher.py`: `find_best_matches` selects a single best semantic match per submitted sentence. It does not group adjacent sentences into passages despite its filename.
- `backend/services/plagiarism_detector.py`: `classify_match` returns `direct_match`, `semantic_match`, `lexical_overlap`, or `low_similarity` using TF-IDF >= 0.70 and SBERT >= 0.75. `detect_candidate_matches` labels every pair but is not used by the main pipeline.
- `backend/services/explanation.py`: `generate_explanation` maps a label to fixed prose and percentage values; `create_match_explanation` shapes an API/storage-ready record.

### Orchestration and persistence

- `backend/services/plagiarism_pipeline.py`
  - `analyze_documents(source_file, submitted_file, minimum_similarity=0.0)` orchestrates the full analysis and returns counts plus selected matches.
  - `save_analysis_result(result, output_file="results/final_analysis.json")` writes a JSON file. It is used by a demo script, not the HTTP endpoint.
- `backend/services/database_service.py`: `save_analysis_to_database` creates one `Analysis`, flushes for its generated ID, inserts a `MatchResult` per result match, commits atomically, rolls back on failure, and closes the session.
- `backend/services/history_service.py`: `get_all_analyses` gives newest-first summary dictionaries; `get_analysis_by_id` returns one analysis and its match dictionaries, or `None`.

## 9. Important Files

| File | Purpose and significant relationships |
| --- | --- |
| `backend/app.py` | Imports pipeline and database/history services; owns all API routes and upload-file writes. |
| `backend/database/database.py` | Defines a relative SQLite URL (`sqlite:///./plagiarism.db`), engine, `SessionLocal`, and shared ORM `Base`. |
| `backend/database/models.py` | Defines the persistent analysis/match schema used by persistence/history services. |
| `backend/services/plagiarism_pipeline.py` | Main reusable composition point for analysis; calls the processing, similarity, selection, classification, and explanation services. |
| `backend/services/sbert_model.py` | Model-loading boundary and only live ML-model provider integration. |
| `frontend/app.py` | Separate visual shell; no HTTP request code exists. |
| `results/*.json` | Sample output produced by demo workflows; neither is read by application code. |
| `requirements*.txt` | Pinned dependency inventories. `requirements-local.txt` differs from `requirements.txt` for several versions and omits/adds transitive packages. |

Empty packages `backend/routes` and `backend/utils` have no current implementation or callers. All listed `__init__.py` files are empty.

## 10. Database / Data Model

SQLite is configured at a working-directory-relative `plagiarism.db`. The file was not present during inspection, consistent with `.gitignore` excluding `*.db`.

```text
analyses
  id (PK, indexed)
  source_document (string, required)
  submitted_document (string, required)
  source_sentence_count (integer, default 0)
  submitted_sentence_count (integer, default 0)
  matched_sentence_count (integer, default 0)
  created_at (datetime, default UTC now)

match_results
  id (PK, indexed)
  analysis_id (FK -> analyses.id, required)
  source_sentence_id (integer, required)
  submitted_sentence_id (integer, required)
  source_page, submitted_page (integer, nullable)
  source_text, submitted_text (text, required)
  tfidf_score, sbert_score (float, required)
  match_type (string, required)
  explanation (text, required)
```

`Analysis.matches` is a one-to-many relationship with `cascade="all, delete-orphan"`; `MatchResult.analysis` is the reverse relationship. There are no schema migrations, uniqueness constraints, explicit match ordering, foreign-key index declaration beyond the primary keys, or user/document tables.

Tables are created only by `tests/test_database.py` through `Base.metadata.create_all(bind=engine)`. The Flask application does not call `create_all`, so a fresh environment will need this initialization before analysis persistence/history succeeds.

## 11. APIs and Integrations

### Internal HTTP API

| Endpoint | Method | Behavior |
| --- | --- | --- |
| `/` | GET | Returns status, API name, and version `1.0`. |
| `/api/health` | GET | Returns `{"status": "healthy"}`; it does not test database/model availability. |
| `/api/analyze` | POST | Requires multipart `source_file` and `submitted_file`; analyzes, persists, and returns counts, paths, `analysis_id`, and matches. Missing/empty file names return 400; exceptions return 500 with `details`. |
| `/api/history` | GET | Returns `{status, total, analyses}` from `get_all_analyses`, newest first. |
| `/api/history/<int:analysis_id>` | GET | Returns a single persisted analysis and all its matches, 404 when absent. |

`/api/analyze` has no authentication, configured file-size limit, allowed-extension precheck, filename sanitization, or cleanup. Filenames are directly joined to `data/uploads`, which means same-name uploads overwrite prior files and client-controlled path-like names are risky. Extraction validates formats later, after the upload has already been written.

### External/model integrations

Only Sentence Transformers is used by source code, with model identifier `all-MiniLM-L6-v2`. There are no tokens, API keys, custom prompts, retries, rate limiting, RAG retrieval, external HTTP calls, or cloud database settings in code. `flask-cors`, `requests`, `httpx`, `uvicorn`, and `starlette` appear in dependency files but are not imported by the application source.

## 12. AI / ML Components

The implementation is a two-signal sentence matching pipeline, not a training or generative-AI system.

- **Lexical signal:** a fresh combined-corpus TF-IDF vectorizer produces pairwise cosine scores. It intentionally uses normalized text only for this comparison.
- **Semantic signal:** SBERT encodes unmodified sentence text with normalized embeddings. Cosine similarity is calculated for each source/submitted pair.
- **Selection:** for each submitted sentence, only the source sentence with maximum SBERT score is retained, subject to `minimum_similarity`.
- **Classification:** thresholds are explicitly called provisional in `plagiarism_detector.py`: 0.70 TF-IDF and 0.75 SBERT.
- **Explanation:** static label-specific prose; no model-generated explanation or citation/retrieval mechanism.

There is no batching policy, evaluation dataset, trained artifact, confidence calibration, model fallback, cache code, or model-load error handling. Importing `sbert_model.py` loads the model immediately, which impacts startup and test imports.

## 13. Configuration

- **Database URL:** hardcoded `DATABASE_URL = "sqlite:///./plagiarism.db"`; the actual location depends on the process working directory.
- **Flask:** binds only to `127.0.0.1:5000`, with `debug=True` when started directly.
- **Uploads:** hardcoded `data/uploads`, created at Flask-module import time.
- **Model:** hardcoded `MODEL_NAME = "all-MiniLM-L6-v2"`.
- **Decision thresholds:** hardcoded in `classify_match`.
- **Environment variables/secrets:** no environment-variable reads, `.env` files, keys, passwords, or tokens were found. No secrets are documented here.

## 14. Dependencies

Important direct dependencies are Flask, Streamlit, SQLAlchemy, PyMuPDF, python-docx, scikit-learn, Sentence Transformers, PyTorch, and Transformers. `requirements.txt` is a broad pinned environment inventory rather than a minimal direct-dependency manifest. `requirements-local.txt` is a second inventory with version differences, including NumPy, pandas, networkx, SciPy, scikit-learn, and `rpds-py`; the repository does not state which is authoritative.

`flask-cors` is installed but no source imports or configures it. Dependencies such as `requests`, `httpx`, `uvicorn`, `starlette`, `reportlab`, and Streamlit visualization packages have no corresponding application-source usage found, although some may be transitive or intended for local tooling.

## 15. Feature Status

| Feature | Status | Evidence | Important files |
| --- | --- | --- | --- |
| PDF/DOCX/TXT text extraction | Implemented | Extension dispatch and extractors exist | `document_processor.py` |
| Sentence-level TF-IDF + SBERT comparison | Implemented | Both matrices are built and example JSON exists | `tfidf_model.py`, `sbert_model.py`, `similarity.py` |
| Explainable candidate labels | Implemented, provisional thresholds | Four labels and template explanations | `plagiarism_detector.py`, `explanation.py` |
| Flask analysis API | Implemented | Five routes, including analysis | `backend/app.py` |
| SQLite persistence/history | Partially implemented operationally | ORM/services exist, but API startup does not create tables | `database/`, `database_service.py`, `history_service.py` |
| Streamlit user workflow | Partially implemented | Static status page only; no uploads/results/history | `frontend/app.py` |
| Automated regression test suite | Unclear / not implemented as assertions | Scripts print output and several write artifacts; no assertions or pytest configuration | `tests/` |
| Passage-level matching | Not implemented | Selection is one source sentence per submitted sentence | `passage_matcher.py` |
| Authentication/authorization | Not implemented | No auth code or configuration found | `backend/` |
| Deployment/CI/CD | Unknown / not determinable from the repository | No relevant config files were found | repository root |

## 16. Recent Local Changes

**Unknown / Not determinable from the repository.** The requested Git inspection could not be performed because this directory has no `.git` metadata and no enclosing Git repository was found. Consequently, no changed/staged/deleted/untracked files or recent commits can be identified, and another agent must not assume the displayed code has a known `HEAD` baseline.

The only new file created for this task is `PROJECT_CONTEXT.md`; no existing project files were changed.

## 17. Known Issues / TODOs

These observations are supported by the code; they are not repairs or guesses.

- The threshold comments explicitly say calibration is future work. The default API threshold of `0.0` means returned matches are candidate best matches, including `low_similarity`, rather than a final positive-plagiarism-only set.
- Empty documents/sentence lists are not validated before vectorization. A combined empty/stopword-like corpus can make `TfidfVectorizer.fit_transform` fail.
- Tables are not initialized by the application; a fresh API analysis request will attempt database writes without any startup `create_all` call.
- Upload filenames are unsanitized and uploads are retained. This permits overwrite on name collision and may permit path traversal depending on the provided filename/platform behavior.
- API exceptions expose `str(e)` to clients, and debug mode is enabled for direct execution.
- `created_at` is returned as a raw Python `datetime` from history services; Flask serialization behavior supplies the actual HTTP representation, but no response schema is defined.
- The Streamlit UI does not invoke the Flask API, so it is not currently an interactive client for the implemented backend.
- `tests/test_api_history_details.py` hardcodes analysis ID `3`, and scripts rely on relative paths and/or existing data. The suite does not verify outcomes with assertions.
- `save_analysis_result` and `tests/test_similarity.py` overwrite files in `results/`; `test_database.py`, `test_database_save.py`, and API analysis can create/write `plagiarism.db` and uploads.

## 18. How to Run

The following commands are inferred from the source and dependency files; they were not run during this inspection because the task restricted repository-modifying operations.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python tests/test_database.py
python backend/app.py
```

The database initialization script is necessary before write/history API use in a fresh working directory. It creates a local `plagiarism.db`, which `.gitignore` excludes. To start the static Streamlit page in another terminal:

```bash
streamlit run frontend/app.py
```

There are no environment variables required by current source. Running the model-based pipeline may cause Sentence Transformers to load or obtain `all-MiniLM-L6-v2` through its normal runtime cache. Do not run write-oriented scripts unless their generated uploads, database, or JSON output are intended.

## 19. Testing

There is no configured test runner or assertion-based test suite. The `tests/` directory is a collection of manual executable checks:

- component demos for extraction, preprocessing, TF-IDF, SBERT, pair classification, matching, and explanation;
- pipeline/similarity scripts that write JSON results;
- database creation/read/save/history scripts;
- Flask test-client scripts for home/health, analysis, and history endpoints.

`python tests/test_flask.py` is read-only aside from import-time creation of `data/uploads`; several other scripts make persistent changes as described above. `pytest` may discover these files, but they execute at import and contain no test functions/assertions, so it should not be treated as a reliable non-mutating test command.

## 20. Deployment

Unknown / Not determinable from the repository. No Dockerfile, process manager configuration, reverse-proxy configuration, cloud manifest, CI/CD workflow, or deployment documentation was found. The only explicit runtime binding is Flask development mode at `127.0.0.1:5000`.

## 21. Architectural Constraints

- Preserve the result dictionary keys consumed by `database_service` and returned from `/api/analyze`; storage assumes every result match has IDs, text, scores, a type, and explanation.
- Preserve the original sentence text and page numbers through preprocessing if evidence display/persistence is needed; TF-IDF normalization is intentionally a separate field.
- The similarity matrix index order must remain aligned with the processed sentence arrays, because `similarity.py` maps matrix coordinates back to those records.
- The model currently loads on module import and uses the fixed SBERT model name; moving this can change startup behavior and callers/tests that import it.
- SQLite is relative to the working directory, so changing how the app is launched can change which database it uses.
- Existing history code expects an `Analysis` relationship named `matches` and accesses match fields directly.
- Any frontend work must explicitly add HTTP communication; no current frontend-backend contract implementation exists in Streamlit.

## 22. Safe Extension Points

- Add new extraction formats or improve extraction under `backend/services/document_processor.py`, keeping the established structured-sentence shape.
- Add model/scoring strategies as new services under `backend/services/`, then compose them through `similarity.py` or `plagiarism_pipeline.py` rather than embedding logic in routes.
- Add API routes in `backend/app.py`; `backend/routes/` is empty, so introducing blueprints there would be a structural change, not continuation of an existing route pattern.
- Add persistence/query operations adjacent to `database_service.py` and `history_service.py`, retaining SQLAlchemy models in `backend/database/models.py`.
- Build actual UI features in `frontend/app.py`, including an explicit Flask HTTP client layer.
- Replace the demo scripts incrementally with isolated, assertion-based tests; avoid using existing sample/output filenames for tests that need isolation.

## 23. Important Relationships Between Files

```text
backend/app.py
  -> plagiarism_pipeline.analyze_documents
       -> document_processor.process_document
       -> preprocessing.prepare_sentences
       -> similarity.calculate_all_similarities
            -> tfidf_model.calculate_tfidf_similarity
            -> sbert_model.calculate_sbert_similarity
       -> passage_matcher.find_best_matches
       -> plagiarism_detector.classify_match
       -> explanation.create_match_explanation
  -> database_service.save_analysis_to_database
       -> database.models (Analysis, MatchResult)
       -> database.database (SessionLocal)
  -> history_service queries the same models/session

tests/ imports these modules directly; it does not call the Streamlit frontend.
frontend/app.py has no imports from backend/ and no HTTP requests.
```

## 24. Development Notes for Future AI Agents

### Before modifying code

Understand that analysis results represent best sentence-level candidates, not necessarily confirmed plagiarism verdicts. Confirm whether a database has been created in the actual runtime directory before diagnosing API history/write failures. Do not infer Git changes from this snapshot; obtain the original repository metadata or a baseline first.

### Existing patterns

Services use plain dictionaries as interchange data, functions are mostly stateless, and pipeline orchestration lives in `plagiarism_pipeline.py`. ORM session lifecycle is local to each service function and uses `try/finally`; write operations add a rollback on exception. The code favors explicit, expanded formatting and comment section dividers.

### Do not duplicate

Reuse `process_document`, `prepare_sentences`, `calculate_all_similarities`, `find_best_matches`, `classify_match`, and `create_match_explanation` rather than reimplementing the analysis chain in routes or UI code. Reuse `save_analysis_to_database` for the existing result shape and the history service for existing history responses.

### Important dependencies

`sbert_model.py` is import-sensitive because it initializes the model globally. `similarity.py` assumes its two matrix functions return identically shaped source-by-submitted arrays. Persistence depends on the exact result match schema created by the explanation module. Flask imports model-dependent pipeline code at application import, so even health-route imports may be affected by model loading.

### Dangerous changes

Changing sentence IDs, page keys, result keys, ORM relationship names, or score semantics affects database writes, history responses, artifacts, and demo scripts. Altering thresholds changes labels in all persisted/future results. Changing SQLite to a different URL alters history visibility. Do not treat the static Streamlit page as already integrated with Flask.

### Extension strategy

Keep processing/model logic in services, persistence logic in services plus database models, and HTTP translation in `backend/app.py`. Add UI behavior separately to `frontend/app.py`. If database schema changes are needed, note that no migration system exists; introduce a deliberate migration/init approach rather than silently relying on `create_all` for existing tables.

### Current unfinished work

Threshold calibration is explicitly pending. The frontend workflow, robust upload handling, DB initialization lifecycle, deployment, authentication, and reliable assertion-based testing are incomplete or absent. Empty `routes` and `utils` packages should not be mistaken for implemented layers.

### User modifications

Unknown / Not determinable from the repository because the expected Git metadata is absent. Preserve all present project files unless a future task explicitly changes them; this context file itself is the sole task-created file.

## 25. Unknown / Uncertain Information

- Project owner, origin, formal project name beyond UI/API strings, license, and intended production use are unknown.
- Git branch, commit history, remotes, baseline, staged/untracked files, and local modifications are not determinable because no Git repository is present.
- The authoritative dependency file is unknown because two differing pinned files exist.
- Whether the SBERT model is already cached, whether model download is permitted, and actual startup performance are unknown without running it.
- A deployment approach, CI, production database, supported Python version, browser clients, and service-level performance limits are not represented in repository configuration.
- The operational quality/coverage of tests is unknown; they were inspected statically and deliberately not run because several modify repository runtime artifacts.
