# Deployment Instructions

This project consists of a FastAPI backend and a Next.js frontend. To deploy or run the project locally with a unified link, follow these steps.

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm

## 1. Start the Backend

The backend runs on port `8000`.

```bash
cd backend
# Install dependencies if needed
pip install -r requirements.txt
# Run the server
python app.py
```

## 2. Start the Frontend

The frontend runs on port `3000` and proxies API requests to the backend.

```bash
cd frontend
# Install dependencies if needed
npm install
# Run the development server
npm run dev
```

## 3. Access the Application

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Using ngrok (Unified Link)

To expose the application with a single link (e.g., for sharing or mobile testing), use ngrok to tunnel port `3000`.

```bash
ngrok http 3000
```

This will provide a public URL (e.g., `https://<random-id>.ngrok-free.app`) that serves both the frontend and the backend API (via the internal proxy).