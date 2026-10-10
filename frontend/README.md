# TextOrigin frontend

Next.js 16.4 / React 19.2. Local API defaults to http://localhost:8000; set NEXT_PUBLIC_TEXTORIGIN_API_URL at build time to override.

Run npm ci and npm run dev. Production: npm run build then npm start.

The UI submits classify, rewrite and compare jobs, polls actual progress, and supports server cancellation. Input/options changes invalidate old results and cancel pending jobs. Keys remain in page memory; output metrics describe accepted text.

Verification: npx playwright install chromium, npm run build, npm test. Browser tests use actual Chromium and clearly simulated API responses, with no model downloads or provider charges. Actual backend/model validation is described in the repository README and docs/TRAINING_EVALUATION.md.
