# DEV Notes

## Web Search Providers

`tools.web.search.provider` รองรับ 4 ค่า: `brave` (default), `tavily`, `gemini`, `grok`

### Brave Search

```json
{
  "tools": { "web": { "search": { "provider": "brave", "apiKey": "BSA..." } } }
}
```

env var: `BRAVE_API_KEY` — [https://brave.com/search/api/](https://brave.com/search/api/)

---

### Tavily

```json
{
  "tools": {
    "web": { "search": { "provider": "tavily", "apiKey": "tvly-..." } }
  }
}
```

env var: `TAVILY_API_KEY` — [https://tavily.com](https://tavily.com)

คืน AI answer + raw search results (title, url, content snippet)

---

### Gemini Grounding

```json
{
  "tools": {
    "web": { "search": { "provider": "gemini", "apiKey": "AIza..." } }
  }
}
```

env var: `GEMINI_API_KEY` — [https://aistudio.google.com/](https://aistudio.google.com/)

ใช้ Gemini + Google Search grounding — คืน AI summary พร้อม sources

ระบุ model ได้ผ่าน `model` (default: `gemini-2.5-flash`):

```json
{
  "tools": {
    "web": {
      "search": {
        "provider": "gemini",
        "apiKey": "AIza...",
        "model": "gemini-2.5-flash"
      }
    }
  }
}
```

---

### Grok Live Search (xAI)

```json
{
  "tools": {
    "web": { "search": { "provider": "grok", "apiKey": "xai-..." } }
  }
}
```

env var: `XAI_API_KEY` — [https://console.x.ai/](https://console.x.ai/)

ใช้ xAI Responses API + web_search tool — คืน AI answer พร้อม citations

ระบุ model ได้ผ่าน `model` (default: `grok-4-1-fast-reasoning`):

```json
{
  "tools": {
    "web": {
      "search": {
        "provider": "grok",
        "apiKey": "xai-...",
        "model": "grok-4-1-fast-reasoning"
      }
    }
  }
}
```
