# ADR-0004: FastAPI backend + Next.js frontend, Streamlit kept as a fallback

* Status: accepted (12 Nov 2026)
* Module: backend and integration / frontend

## Context

The original prototype was a single Streamlit app that loaded the model itself. The project needs a clear
separation between model serving and UI, a documented API, proper input validation and a polished UI.

## Decision

* **Backend:** FastAPI with a layered structure (`api/`, `core/`, `schemas/`, `services/`), Pydantic v2
  schemas (which also generate the OpenAPI docs), settings from environment variables, the model loaded once
  in the lifespan handler, and one error format for 400/413/415/422/503.
* **Frontend:** Next.js (App Router) + TypeScript + Tailwind + shadcn/ui, calling the API from the browser.
* **Streamlit dashboard** kept as a research fallback; it uses the same `ml.inference.Segmenter` so its
  numbers match the API exactly.

## Consequences

* The model is served by one tested component (23 API tests), and any client can use it.
* CORS must list the frontend origin (`FETALAI_CORS_ORIGINS`).
* Two runtimes (Python and Node) to install; Docker Compose is the planned way to simplify this.
