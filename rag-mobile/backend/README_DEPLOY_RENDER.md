# 🚀 Step-by-Step Guide: Deploying Backend to Render

This guide walks you through deploying the **Pakistan Mobile Phone Shopping Assistant Backend** to **Render (Free Tier)**.

---

## 📋 Prerequisites

Before starting, ensure you have:
1. **A GitHub Account** ([github.com](https://github.com)) with this project pushed to a repository.
2. **A Render Account** ([render.com](https://render.com)) — Free tier is completely sufficient.
3. **API Keys**:
   - **Pinecone API Key**: From [app.pinecone.io](https://app.pinecone.io)
   - **Groq API Key**: From [console.groq.com/keys](https://console.groq.com/keys)
   - *(Optional)* **Gemini API Key**: From [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
   - *(Optional)* **HuggingFace Token**: From [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

> [!IMPORTANT]
> Make sure you have run the local ingestion pipeline **at least once** before deploying so your Pinecone index has all 2,790 phone vectors:
> ```bash
> cd ingestion_pipeline
> pip install -r requirements.txt
> python run_ingestion.py
> ```

---

## 🛠️ Method 1: Web Service Setup (Recommended & Simplest)

### Step 1: Push Your Code to GitHub

Open your terminal in the project root and commit/push your changes:

```bash
git add .
git commit -m "Optimize backend for Render and update frontend UI"
git push origin main
```

---

### Step 2: Create a New Web Service on Render

1. Log into your **[Render Dashboard](https://dashboard.render.com/)**.
2. Click the **"New +"** button in the top navigation bar and select **"Web Service"**.
3. Choose **"Build and deploy from a Git repository"** and click **Next**.
4. Connect your GitHub repository (`rag-mobile-assistant` or your repo name).

---

### Step 3: Configure Web Service Settings

Fill in the settings form with the following exact values:

| Field | Value | Notes |
|---|---|---|
| **Name** | `mobile-rag` | *(If you use `mobile-rag`, your URL will be `https://mobile-rag.onrender.com` which matches the frontend default)* |
| **Region** | `Frankfurt (EU Central)` or `Oregon (US West)` | Pick the region closest to you |
| **Branch** | `main` | Or your default branch name |
| **Root Directory** | `rag-mobile/backend` *(or `backend` if at root)* | Path to the backend directory containing `requirements.txt` and `app/` |
| **Runtime** | `Python 3` | |
| **Build Command** | `pip install -r requirements.txt` | Installs slim runtime dependencies (~15MB) |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` | Starts FastAPI with dynamic Render port |
| **Instance Type** | `Free` (0.1 CPU, 512 MB RAM) | $0/month free tier |

---

### Step 4: Add Environment Variables

Scroll down to the **"Environment Variables"** section and add the following keys:

#### 🔹 Required Variables:

| Key | Value | Description |
|---|---|---|
| `PINECONE_API_KEY` | `pcsk_...` *(your actual key)* | Pinecone vector DB key |
| `PINECONE_INDEX_NAME` | `mobile-rag` | Name of your Pinecone index |
| `PINECONE_CLOUD` | `aws` | Pinecone cloud provider |
| `PINECONE_REGION` | `us-east-1` | Pinecone region |
| `PINECONE_NAMESPACE` | `mobiles` | Namespace where vectors are stored |
| `MEMORY_BACKEND` | `memory` | Uses in-RAM session memory (resets on restart, zero disk usage) |
| `LLM_PROVIDER` | `groq` | Primary LLM provider |
| `GROQ_API_KEY` | `gsk_...` *(your actual key)* | Groq API key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Default Groq model |
| `TOP_K` | `5` | Number of phone specs retrieved per query |
| `MAX_HISTORY_TURNS` | `10` | Chat conversation turn window |

#### 🔹 Optional Variables:

| Key | Value | Description |
|---|---|---|
| `GEMINI_API_KEY` | `AQ...` | Enables Google Gemini option in UI |
| `GEMINI_MODEL` | `gemini-2.0-flash` | Gemini model name |
| `MISTRAL_API_KEY` | `...` | Enables Mistral option in UI |
| `MISTRAL_MODEL` | `mistral-small-latest` | Mistral model name |
| `HUGGINGFACE_API_KEY` | `hf_...` | HuggingFace token (for higher embedding rate limits) |

---

### Step 5: Deploy & Monitor

1. Click **"Create Web Service"** (or **"Deploy"**).
2. Render will pull your repo, run `pip install -r requirements.txt`, and start Uvicorn.
3. Look for the following in the build logs:
   ```text
   ==> Uploading build...
   ==> Build successful 🎉
   ==> Starting service with 'uvicorn app.main:app --host 0.0.0.0 --port $PORT'
   ============================================================
     Pakistan Mobile Assistant — Backend Starting
     Memory backend : MEMORY
     CORS origins   : ['*']
     Pinecone index : ✅ OK (2790 vectors)
   ============================================================
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:10000
   ```

---

## 🔍 Step 6: Verify Backend Health

Once the status turns to **"Live"**, open your browser and navigate to:

```
https://<YOUR-RENDER-SERVICE-NAME>.onrender.com/health
```

You should receive a JSON response like this:

```json
{
  "status": "ok",
  "indexed_chunks": 2790,
  "memory_backend": "memory"
}
```

You can also check the interactive Swagger API documentation at:
```
https://<YOUR-RENDER-SERVICE-NAME>.onrender.com/docs
```

---

## 🌐 Step 7: Connecting with Frontend

1. If your Render service name is **`mobile-rag`**, its URL is `https://mobile-rag.onrender.com`.
2. This URL is already preset in [`frontend/index.html`](file:///g:/rag-mobile-assistant/rag-mobile-assistant/rag-mobile/frontend/index.html#L555):
   ```javascript
   const BACKEND_URL = 'https://mobile-rag.onrender.com';
   ```
3. If you chose a different service name (e.g. `https://my-app.onrender.com`), update `const BACKEND_URL` in `frontend/index.html` to your Render URL and push/redeploy the frontend to Netlify/Vercel.

---

## 💡 Important Things to Know About Render Free Tier

1. **Cold Starts (Spin Down)**:
   - On the Free plan, Render puts services to sleep after **15 minutes of inactivity**.
   - When a user visits the chat app after it has been sleeping, the first request will take **~30 to 50 seconds** to spin back up.
   - The frontend displays a helpful friendly status indicator (`● AI Online` / `● AI Offline`) to inform users.

2. **Temporary Chat Memory**:
   - Because `MEMORY_BACKEND=memory` is enabled, chat history is held in server RAM during active conversations.
   - When Render spins down or restarts, chat sessions reset cleanly without leaving corrupt state or filling up disk space.

3. **No Heavy ML on Server**:
   - Query embedding is performed via HuggingFace's hosted API (`sentence-transformers/all-MiniLM-L6-v2`), consuming virtually zero server RAM and staying well within the 512MB RAM free limit.
