# TextOrigin frontend

This Next.js App Router client provides classification and humanization controls for the TextOrigin FastAPI backend.

## Setup

```bash
npm ci
copy .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. The API defaults to `http://localhost:8000`; override it at build time with:

```text
NEXT_PUBLIC_TEXTORIGIN_API_URL=https://textorigin-api.example.com
```

`NEXT_PUBLIC_` values are visible in the browser bundle and must not contain secrets.

## Structure

- `src/app/page.js`: page-level input and classification state
- `src/components/ClassificationResults.js`: score and observation rendering
- `src/components/HumanizePanel.js`: humanization interaction and results
- `src/lib/api.js`: HTTP contract, timeout, and error normalization

## Verification

```bash
npm run build
```

The backend must separately pass `pytest -q`. The application deliberately uses local component state; no global state library is needed for the current workflow.
