# UniBuddy

UniBuddy is a full‑stack web application that helps students connect, share information, and stay organized. It includes a React + Vite frontend and a Python backend, all wired together with a one‑command launcher.

---

## Features

- Modern frontend (React + Vite + TypeScript)
- Python backend API (runs on port 9000)
- One‑command start script that runs frontend and backend together
- Environment‑based configuration via `.env` files
- Hot‑reload dev experience for quick iteration

---

## Project Structure

```text
UniBuddy/
  backend/        # Python backend (APIs, DB, business logic)
  public/         # Static assets served by frontend
  src/            # Frontend source (React components, hooks, etc.)
  start_all.py    # Helper script to run frontend + backend together
  package.json    # Frontend dependencies & scripts
  tsconfig*.json  # TypeScript configuration
  vite.config.ts  # Vite config
```

---

## Getting Started

### Prerequisites

- **Node.js** (LTS recommended)
- **Python 3.10+** (or compatible with your backend)
- **Git**

### 1. Install Frontend Dependencies

From the project root:

```bash
cd UniBuddy
npm install
```

### 2. Configure Environment Variables

Create a `.env` file in the project root (if it doesn’t already exist):

```bash
# example .env
VITE_API_BASE_URL=http://localhost:9000
BACKEND_PORT=9000
NODE_ENV=development
```

> Never commit real secrets (API keys, DB passwords, etc.). They are already ignored via `.gitignore`.

---

## One‑Command Launch (Recommended)

From the project root, run:

```bash
python start_all.py
```

This will:

- Automatically install frontend dependencies if needed
- Start both backend (port 9000) and frontend dev server
- Show interleaved logs with clear prefixes
- Shut down both cleanly with `Ctrl+C`

Once everything is running:

- Frontend: `http://localhost:5173` (default Vite port)
- Backend API: `http://localhost:9000`

---

## Running Frontend / Backend Separately

### Frontend only

```bash
npm run dev
```

### Backend only

From the `backend` folder (example, adjust to your entry file):

```bash
cd backend
python main.py
```

---

## Available NPM Scripts

Common scripts (see `package.json` for the full list):

```bash
npm run dev       # Start Vite dev server
npm run build     # Production build
npm run preview   # Preview production build locally
npm run lint      # Run linting
```

---

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is currently unlicensed. If you plan to use it in production or as open source, consider adding a license (e.g., MIT, Apache 2.0) in `LICENSE`.
