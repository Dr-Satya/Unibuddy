# UniBuddy — AI-Powered University Assistant for GD Goenka University

A full-stack web application that combines JWT-based authentication with a RAG-powered chatbot to answer student queries about timetables, faculty, mentor assignments, fees, and general university information.

---

## What It Does Right Now

- Students and admins authenticate via a secure Node.js/MongoDB backend
- Authenticated users interact with a floating chatbot widget powered by FAISS + Groq LLaMA
- The chatbot handles three distinct query types: guided timetable lookup, mentor-mentee search, and general RAG-based Q&A
- Admins can manage student records, upload PDF timetables, and upload Excel mentor-mentee sheets that directly power chatbot answers

---

## Three Services

| Service | Stack | Port |
|---|---|---|
| Frontend | React 19 + TypeScript + Vite | 5173 |
| Auth Backend | Node.js + Express + MongoDB | 5000 |
| Chatbot Backend | Python + FastAPI + Groq + FAISS | 9000 |

---

## Prerequisites

- Node.js v18+
- Python 3.10+
- MongoDB (local or Atlas)
- A [Groq API key](https://console.groq.com/) (free tier works)
- A [Mailtrap](https://mailtrap.io/) token for email verification

---

## Setup

### 1. Frontend

```bash
# From project root
npm install
```

Create no extra `.env` — the frontend talks to `localhost:5000` and `localhost:9000` directly.

### 2. Auth Backend

```bash
cd backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend
npm install
```

Create `.env` in that folder:

```env
MONGO_URI=mongodb://localhost:27017/unibuddy
JWT_SECRET=your_jwt_secret_here
CLIENT_URL=http://localhost:5173
MAILTRAP_TOKEN=your_mailtrap_token_here
PORT=5000
```

### 3. Chatbot Backend

```bash
cd backend
pip install -r requirements.txt
```

Create `.env` in `backend/`:

```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=unibuddy_dev_secret
VECTOR_DB_PATH=./data/vectordb/
TOP_K_RESULTS=15
DEBUG_RAG=false
```

---

## Running the App

Open three terminals:

```bash
# Terminal 1 — Frontend
npm run dev

# Terminal 2 — Auth backend
cd backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend
nodemon index.js

# Terminal 3 — Chatbot backend
cd backend
uvicorn api:app --host 0.0.0.0 --port 9000
```

Then open **http://localhost:5173**

---

## Architecture

```
Browser (localhost:5173)
         |
         |--- React App (Vite)
         |     |--- Zustand authStore (JWT state)
         |     |--- AuthModalContext (login/signup modals)
         |     |--- Chatbot.tsx (floating widget, all pages)
         |     |--- AdminPanel.tsx (students + timetable + mentor tabs)
         |
         |  [1] Auth requests          [2] Chat messages
         v                              v
Node.js / Express               Python / FastAPI
(localhost:5000)                (localhost:9000)
    |                                  |
    |--- /api/auth/*                   |--- POST /chat
    |--- /api/students (CRUD)          |       |
    |--- /api/users                    |       |-- Timetable path
         |                             |       |   timetable_store.py
         v                             |       |   --> Groq LLaMA
     MongoDB                           |       |
     (users + students)                |       |-- Mentor path
                                       |       |   mentor_store.py
                                       |       |   (pure Python, no LLM)
                                       |       |
                                       |       '-- RAG path
                                       |           threaded_rag.py
                                       |           --> FAISS index
                                       |           --> Groq LLaMA
                                       |
                                       |--- POST /timetable
                                       |     pdfplumber -> timetables.json
                                       |
                                       '--- POST /mentor-mentee
                                             openpyxl -> mentor_mentee.json

Data files (backend/data/):
  vectordb/index.faiss      pre-built FAISS vector index
  vectordb/metadata.json    chunk metadata for index
  vectordb/extra_*.json     per-entity knowledge chunks (auto-saved)
  timetables.json           parsed timetable data (admin-uploaded)
  mentor_mentee.json        mentor-student assignments (admin-uploaded)
  timetable_sessions.json   per-session timetable conversation state
```

---

## Chatbot Query Routing

Every message sent to `/chat` is routed through this priority chain in `api_adapter.py`:

```
Incoming message
    |
    |-- Active timetable session? (mid Year/Branch/Section flow)
    |       YES --> timetable_store.py continues the flow
    |
    |-- Mentor intent keywords? (mentor, mentee, registration, enrollment...)
    |       YES --> mentor_store.py (no LLM, pure lookup)
    |
    |-- Timetable intent keywords? (class, schedule, today, monday...)
    |       YES --> timetable_store.py starts Year/Branch/Section flow
    |
    '-- Everything else
            --> FAISS semantic search (all-MiniLM-L6-v2, top-15)
            --> Groq llama-3.1-8b-instant
            --> Structured HTML reply
```

---

## AI / RAG Pipeline

**Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)

**Vector DB:** FAISS `IndexFlatL2` loaded from `data/vectordb/index.faiss`

**Retrieval:** 3 query variants sent per message (raw, `+ profile`, `+ research`) — top-15 chunks each, deduped before combining

**LLM:** Groq API — `llama-3.1-8b-instant`
- Timetable queries: temp=0.0, max_tokens=400, strict day-column system prompt
- General RAG: temp=0.1, max_tokens=800, university assistant system prompt

**Conversation memory:** Last 10 turns stored in-memory per session ID (Python dict, not persisted to DB)

---

## Admin Panel

Accessible to ADMIN-role users at `/admin`. Three tabs:

**Students tab**
- View all students with search, filter by status, pagination (10 per page)
- Add / edit / delete student records
- Export visible list as CSV
- Each student: name, email, roll number, department, year, contact, photo, college ID card

**Timetable tab**
- Drag-and-drop or click to upload a GD Goenka timetable PDF
- `pdfplumber` extracts class names, time slots, and subject-room-faculty text per day using bounding-box word positioning
- Parsed data saved to `data/timetables.json` — immediately queryable by chatbot
- Preview grid and JSON download available in the UI

**Mentor-Mentee tab**
- Upload `.xlsx`, `.xls`, or `.csv` mentor-mentee allocation sheet
- Flexible column name normalization (50+ aliases supported — "student registration id", "roll no", "enrollment no" all map to the same field)
- Saved to `data/mentor_mentee.json` — immediately queryable by chatbot
- Students can then ask: "Who is my mentor?", "Show mentees of Dr. Mehta", "Details of registration 230160203052"

---

## Authentication Flow

```
Signup --> MongoDB save --> Mailtrap OTP email --> /verify-email code entry --> verified

Login --> bcrypt verify --> JWT in HTTP-only cookie --> Zustand authStore updated

checkAuth() on every page load --> GET /api/auth/check-auth --> restores session from cookie
```

**Roles:**
- `STUDENT` — chatbot access only
- `ADMIN` — chatbot + admin panel (CRUD, uploads)

Admin role is assigned at signup if the email is in `config/adminWhitelist.js`.

---

## API Endpoints

### Auth Backend (port 5000)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/signup` | Register (email, password, names, photo, ID card) |
| POST | `/api/auth/login` | Login → sets JWT cookie |
| POST | `/api/auth/logout` | Clears JWT cookie |
| POST | `/api/auth/verify-email` | Verify 6-digit OTP |
| POST | `/api/auth/forgot-password` | Send reset link via email |
| POST | `/api/auth/reset-password/:token` | Reset password with token |
| GET | `/api/auth/check-auth` | Returns current user from cookie |
| GET | `/api/students` | List all students (admin only) |
| POST | `/api/students` | Add student (admin only) |
| PUT | `/api/students/:id` | Update student (admin only) |
| DELETE | `/api/students/:id` | Delete student (admin only) |

### Chatbot Backend (port 9000)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat` | Send message `{ message, session_id, user_profile? }` |
| POST | `/timetable` | Upload timetable PDF (multipart) |
| POST | `/mentor-mentee` | Upload mentor-mentee Excel/CSV (multipart) |
| GET | `/sections` | Get available section options for signup dropdown |
| GET | `/health` | Health check |

---

## Project Structure

```
UniBuddy/
|-- src/                          Frontend (React + TypeScript)
|   |-- App.tsx                   Routes + auth check on load
|   |-- main.tsx                  React entry point
|   |-- store/
|   |   '-- authStore.ts          Zustand auth state + all auth actions
|   |-- context/
|   |   '-- AuthModalContext.tsx  Global modal state (login/signup/etc.)
|   |-- components/
|   |   |-- Chatbot.tsx           Floating chatbot widget (all pages)
|   |   |-- AuthModal.tsx         Modal wrapper (backdrop + animation)
|   |   |-- AuthModalManager.tsx  Renders correct form inside modal
|   |   |-- TimetableUpload.tsx   PDF upload + grid preview
|   |   |-- MentorUpload.tsx      Excel/CSV upload UI
|   |   |-- Navbar.tsx            Top nav with login/signup buttons
|   |   |-- ProtectedRoute.tsx    Route guard (auth + role check)
|   |   '-- RedirectAuthenticated Redirects logged-in users from /login
|   '-- pages/
|       |-- HomePage.tsx          Landing page (Navbar + Hero + Features)
|       |-- LoginPage.tsx         Login form (rendered inside modal)
|       |-- SignupPage.tsx        Signup form (rendered inside modal)
|       |-- EmailVerification...  OTP input page
|       |-- ForgotPasswordPage    Forgot password form
|       |-- ResetPasswordPage     Reset password form
|       |-- AdminPanel.tsx        Full admin UI (3 tabs)
|       '-- DashboardPage.tsx     Profile page (exists, not routed yet)
|
|-- backend/
|   |-- api.py                    FastAPI entry point (all routes)
|   |-- requirements.txt          Python dependencies
|   |-- .env                      GROQ_API_KEY, SECRET_KEY, etc.
|   |-- tests/
|   |   '-- test_api.py           Unit + integration tests for /chat endpoint
|   |-- data/
|   |   |-- vectordb/             FAISS index + metadata + extra chunks
|   |   |-- timetables.json       Admin-uploaded timetable data
|   |   |-- mentor_mentee.json    Admin-uploaded mentor data
|   |   '-- timetable_sessions    Per-session guided flow state
|   '-- src/
|       |-- api_adapter.py        Core request router + session history
|       |-- threaded_rag.py       FAISS vector search
|       |-- threaded_models.py    Groq model manager
|       |-- timetable_store.py    Timetable session flow + LLM answering
|       |-- mentor_store.py       Mentor-mentee lookup + HTML rendering
|       |-- timetable_lookup.py   Section-key-based lookup (secondary path)
|       '-- config.py             All settings (env-var backed)
|
'-- backend/Unibuddy-Auth/
    '-- Unibuddy-MERN-Authentication/backend/
        |-- index.js              Express entry point
        |-- .env                  MONGO_URI, JWT_SECRET, MAILTRAP_TOKEN
        |-- routes/               auth + student + user routes
        |-- controllers/          auth + user business logic
        |-- models/               User + Student Mongoose schemas
        |-- middleware/           verifyToken JWT middleware
        '-- config/
            '-- adminWhitelist.js Admin email list
```

---

## Testing

The `backend/tests/` directory contains API and integration tests.

**Run the unit smoke test (no server needed):**
```bash
cd backend
python -m pytest tests/test_api.py::test_get_reply_local -v
```

**Run the live API test (server must be running on port 9000):**
```bash
# Set the API key env var first
set UNIBUDDY_API_KEY=your_key_here        # Windows CMD
$env:UNIBUDDY_API_KEY = "your_key_here"   # Windows PowerShell

cd backend
python -m pytest tests/test_api.py::test_api_chat_endpoint -v
```

`test_get_reply_local` imports `get_reply` directly from `src.api_adapter` — it tests the full RAG pipeline in-process without an HTTP round-trip.

`test_api_chat_endpoint` hits the live `/chat` endpoint. Set `UNIBUDDY_API_URL` to override the default `http://127.0.0.1:9000`.

---

## FAISS Knowledge Base — Current State

The vector index at `backend/data/vectordb/index.faiss` currently contains:

| Metric | Value |
|---|---|
| Total vectors | 11,065 |
| Unique source URLs | 85 |
| Faculty profile pages indexed | 71 (School of Engineering & Sciences only) |
| Fee structure chunks | 466 |
| Admissions page chunks | 457 |
| Course detail chunks | 1,052 |
| About / academics / facilities | 1,433 |

All 71 faculty profiles are from `/school-of-engineering/` URLs only. Faculty from other schools (management, law, hospitality, etc.) are not indexed.

**Known issue — duplicate vectors:** Each faculty profile page has been embedded multiple times (5 copies on average), so 73–80% of the top-15 retrieved chunks for any faculty query are exact duplicates of the same content. The chatbot still answers correctly but wastes most of its retrieval budget. The cause is that `scrape_all_faculty.py` was run multiple times without clearing the index first — `faiss.IndexFlatL2.add()` appends unconditionally and there is no deduplication check anywhere in the ingestion pipeline.

---

## Known Issues / Incomplete Features

- `DashboardPage.tsx` — built but not registered in any route in `App.tsx` (unreachable)
- `ChatbotPage.tsx` — built but not registered in `App.tsx` routes (unreachable)
- `backend/data/fees_data.json` — fee data exists but is not indexed into FAISS and is not loaded by any active code path
- `intent_parser.py` — LLM-based intent parser written but never called in the active request chain
- Google OAuth stubs in `google_auth.py` — router exists but is not mounted in `api.py`
- Chatbot backend has no auth middleware — the frontend enforces auth client-side only
- `scraper.py` / `threaded_scraper.py` — university web scrapers written but not wired into any endpoint
- `services.py`, `threaded_services.py`, `auth.py` (Python) — complete implementations not used in the active request path
- Navbar links "Documentation", "Security", "Campus Map" are decorative only — no hrefs
- FAISS index contains ~5× duplicate vectors per faculty page — retrieval quality is degraded (2.0–3.3/10 on test queries)

---

## Troubleshooting

**Auth backend won't start**
- Check MongoDB is running and `MONGO_URI` in `.env` is correct
- Check no other process is on port 5000: `netstat -ano | findstr :5000`

**Chatbot backend won't start**
- Check `GROQ_API_KEY` is set in `backend/.env`
- Install dependencies: `pip install -r requirements.txt`
- If FAISS index is missing, the RAG path will return empty context but still respond via Groq

**Email verification not sending**
- Check `MAILTRAP_TOKEN` in the auth backend `.env`
- Mailtrap free tier limits apply — check your inbox at mailtrap.io

**Admin role not assigned**
- Email must be in `config/adminWhitelist.js` before signup
- Restart the auth backend after modifying the whitelist

**Timetable not answering after upload**
- Verify the PDF is a GD Goenka timetable format (the parser uses bounding-box word positions)
- Check `backend/data/timetables.json` was written after upload
- Try re-uploading — each upload replaces the existing data

**Mentor queries returning "no data"**
- Upload a mentor-mentee Excel/CSV from Admin Panel → Mentor-Mentee tab first
- Check `backend/data/mentor_mentee.json` exists after upload

---

## Tech Stack

**Frontend:** React 19, TypeScript, Vite, Tailwind CSS, Zustand, React Router v7, Framer Motion, Axios, Lucide React

**Auth Backend:** Node.js, Express, MongoDB/Mongoose, JWT (HTTP-only cookies), bcryptjs, Mailtrap

**Chatbot Backend:** Python, FastAPI, Groq API (llama-3.1-8b-instant), FAISS, sentence-transformers (all-MiniLM-L6-v2), pdfplumber, openpyxl, python-dotenv

---

## Admin Whitelist

Admin role is granted to emails listed in:
```
backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/config/adminWhitelist.js
```

Add an email to this file and restart the auth backend to grant admin access on next signup.

---

*Built for GD Goenka University*
