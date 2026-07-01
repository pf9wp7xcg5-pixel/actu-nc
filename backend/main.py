import feedparser
import httpx
import asyncio
import time
import re
import os
import json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

app = FastAPI(title="Actu NC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Sources ────────────────────────────────────────────────────────────────────
DEFAULT_SOURCES = [
    {"id": "radiococotier", "name": "Radio Cocotier",              "tag": "politique", "url": "https://radiococotier.nc/feed"},
    {"id": "lnc",           "name": "Les Nouvelles Calédoniennes", "tag": "politique", "url": "https://www.lnc.nc/rss.xml"},
    {"id": "voixducaillou", "name": "La Voix du Caillou",         "tag": "politique", "url": "https://voixducaillou.nc/feed"},
    {"id": "dnc",           "name": "Demain NC",                  "tag": "politique", "url": "https://www.dnc.nc/feed"},
    {"id": "noumeapost",    "name": "Nouméa Post",                "tag": "politique", "url": "https://noumeapost.nc/feed"},
    {"id": "gouvnc",        "name": "Gouvernement NC",            "tag": "politique", "url": "https://gouv.nc/actualites/rss.xml"},
    {"id": "neotech",       "name": "NeoTech NC",                 "tag": "tech",      "url": "https://neotech.nc/feed"},
    {"id": "lnceco",        "name": "LNC Économie",               "tag": "eco",       "url": "https://www.lnc.nc/economie/rss.xml"},
    {"id": "lncpolitique",  "name": "LNC Politique",              "tag": "politique", "url": "https://www.lnc.nc/politique/rss.xml"},
]

SOURCES_FILE = Path(__file__).parent / "sources.json"

def _load_sources() -> list[dict]:
    if SOURCES_FILE.exists():
        try:
            return json.loads(SOURCES_FILE.read_text())
        except Exception as e:
            print(f"[sources] erreur lecture sources.json: {e}")
    return list(DEFAULT_SOURCES)

SOURCES: list[dict] = _load_sources()

# ── Cache ──────────────────────────────────────────────────────────────────────
_cache: dict = {"articles": [], "ts": 0}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Helpers ────────────────────────────────────────────────────────────────────
def strip_html(text: str) -> str:
    if not text:
        return ""
    clean = re.sub(r"<[^>]+>", " ", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean[:800]


def parse_date(entry) -> str:
    for field in ("published_parsed", "updated_parsed"):
        t = getattr(entry, field, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc).isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()


async def fetch_feed(source: dict, client: httpx.AsyncClient) -> list[dict]:
    try:
        resp = await client.get(source["url"], timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        feed = feedparser.parse(resp.text)
        articles = []
        for entry in feed.entries[:10]:
            raw = strip_html(
                getattr(entry, "summary", None)
                or getattr(entry, "description", None)
                or getattr(entry, "content", [{}])[0].get("value", "")
            )
            articles.append({
                "id":      entry.get("id", entry.get("link", "")),
                "source":  source["name"],
                "tag":     source["tag"],
                "title":   entry.get("title", ""),
                "url":     entry.get("link", ""),
                "excerpt": raw[:500],
                "date":    parse_date(entry),
                "summary": None,
            })
        return articles
    except Exception as e:
        print(f"[feed error] {source['name']}: {e}")
        return []


async def summarize_groq(excerpt: str, title: str, groq_key: str) -> str:
    """Génère un résumé court en français (les articles sont déjà en français)."""
    if not groq_key or not excerpt.strip():
        return ""
    prompt = (
        f"Article NC : {title}\n\n{excerpt}\n\n"
        "Résume en 2 phrases courtes et factuelles en français. "
        "Réponds UNIQUEMENT avec le résumé, sans introduction."
    )
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 120,
                    "temperature": 0.2,
                },
            )
            data = resp.json()
            if "choices" not in data:
                return ""
            return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[groq error] {e}")
        return ""


async def refresh_cache():
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = [fetch_feed(s, client) for s in SOURCES]
        results = await asyncio.gather(*tasks)

    articles: list[dict] = []
    for batch in results:
        articles.extend(batch)

    # Dédupliquer par URL
    seen = set()
    deduped = []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            deduped.append(a)
    articles = deduped

    articles.sort(key=lambda a: a["date"], reverse=True)

    # Résumés Groq sur les 40 premiers articles si clé présente
    if GROQ_API_KEY:
        sem = asyncio.Semaphore(3)

        async def safe_summarize(a):
            async with sem:
                text = a["excerpt"] or a["title"]
                if text and len(text.strip()) > 50:
                    a["summary"] = await summarize_groq(text, a["title"], GROQ_API_KEY)
            return a

        first_batch = articles[:40]
        rest = articles[40:]
        summarized = await asyncio.gather(*[safe_summarize(a) for a in first_batch])
        articles = list(summarized) + rest

    _cache["articles"] = articles
    _cache["ts"] = time.time()
    print(f"[cache] {len(articles)} articles chargés")


# ── Routes ─────────────────────────────────────────────────────────────────────
scheduler = AsyncIOScheduler()

@app.on_event("startup")
async def startup():
    await refresh_cache()
    # Refresh toutes les 2h (actus locales changent souvent)
    scheduler.add_job(
        refresh_cache,
        CronTrigger(minute=0, hour="*/2", timezone="UTC"),
        id="refresh",
        replace_existing=True,
    )
    scheduler.start()
    print("[scheduler] Refresh toutes les 2h")

@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()


@app.get("/api/feed")
async def get_feed(
    tag:   Optional[str] = Query(None),
    q:     Optional[str] = Query(None),
    limit: int           = Query(100, le=300),
):
    articles = _cache["articles"]

    if tag and tag != "all":
        articles = [a for a in articles if a["tag"] == tag]

    if q:
        q_lower = q.lower()
        articles = [
            a for a in articles
            if q_lower in a["title"].lower() or q_lower in (a["excerpt"] or "").lower()
        ]

    return {
        "articles":  articles[:limit],
        "total":     len(articles),
        "cached_at": datetime.fromtimestamp(_cache["ts"], tz=timezone.utc).isoformat() if _cache["ts"] else None,
    }


@app.get("/api/sources")
async def get_sources():
    return {"sources": SOURCES}


@app.get("/api/refresh")
async def manual_refresh():
    await refresh_cache()
    return {"ok": True, "articles": len(_cache["articles"])}


@app.get("/health")
async def health():
    return {"status": "ok", "articles": len(_cache["articles"]), "cached_at": _cache["ts"]}
