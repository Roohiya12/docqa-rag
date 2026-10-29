# Deploy for free on Streamlit Community Cloud

Result: a public link anyone can open, backed by your Groq free-tier key (kept secret).

## 1. Push the project to a public GitHub repo
```bash
git status            # confirm .env is NOT listed
git add .
git commit -m "Deploy-ready: demo mode, tests"
git push
```
On github.com open your repo and check that `.env` is not in the file list.

## 2. Create the app
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** (label may differ slightly) and pick your repo.
3. Branch: `main`. Main file path: `app/streamlit_app.py`.
4. **Advanced settings** -> Python version **3.11**.
5. In the **Secrets** box paste (TOML format, with quotes):
```toml
GROQ_API_KEY = "paste-your-key-here"
DEMO_MODE = "1"
```
6. Click **Deploy**. The first load builds the index from the sample documents, so it is slower than later loads.

## 3. What demo mode does
| Setting | Effect |
|---|---|
| `DEMO_MODE = "1"` | 15 questions and 5 indexing runs per visitor session; the Rebuild button is hidden |
| `DEMO_MAX_QUESTIONS = "10"` (optional) | Change the per-session question cap |
| `MAX_UPLOAD_CHARS = "150000"` (optional) | Total text allowed per visitor's uploads |
| `GROQ_CHAT_MODEL` / `GROQ_EMBED_MODEL` (optional) | Override model IDs if Groq renames them |

**Uploads work on the public app.** Each visitor's files are parsed and indexed in that visitor's own session memory. They are never written to disk and never visible to other visitors, and they disappear when the session ends.

The hosted disk is temporary: the sample index is rebuilt automatically after a restart.

## 4. Test it
- `What was Solstice's total revenue in fiscal year 2025?` -> $482 million with a `[1]` citation
- `Who is the CEO of Google?` -> should say the documents don't contain it

## 5. Troubleshooting
| Symptom | Fix |
|---|---|
| `unsupported version of sqlite3` | Confirm Python 3.11 in Advanced settings, then Reboot app. If it persists, replace ChromaDB with a small in-memory store (ask for this). |
| `GROQ_API_KEY is not set` | Secrets box: key name must be exactly `GROQ_API_KEY`, value in quotes |
| `model_not_found` / 404 | Run `python scripts/list_groq_models.py` locally; put valid IDs in Secrets |
| `429` / rate limit | Wait a minute; the client retries automatically. Avoid the re-rank checkbox on the public app |
| App asleep | Free apps may sleep when unused; opening the link wakes them |

## 6. Security
- The key lives only in Secrets, never in the repo. If it leaks, delete it at https://console.groq.com/keys and create a new one.
- Every visitor question spends your free quota. The per-session cap limits this but does not stop determined abuse.

## 7. Finish
Add the live URL and a screenshot to the top of your README.
