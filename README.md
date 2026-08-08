# 🌦️ Ensemble Weather

An honest weather app that shows **model disagreement as uncertainty**.

Most weather apps show you one number. This app queries multiple independent weather models (GFS, ECMWF, ICON, GEM, MetNo) at your exact GPS location and shows you:

- The **mean forecast** across all models
- Where they **agree** (high confidence) vs **disagree** (low confidence)
- Each model's individual prediction, so you can judge for yourself

When models diverge — typically near storms, fronts, or complex terrain — confidence drops and you know to trust the forecast less.

## Stack

- Single static HTML file, no build step, no dependencies
- Data via [Open-Meteo](https://open-meteo.com) (free, no API key)
- Deployed on Cloudflare Pages

## Roadmap

- **v1 (this)** — multi-model ensemble spread as uncertainty
- **v2** — cross-check against nearest real ASOS/METAR station (Synoptic Data)
- **v3** — radar overlay + short-term nowcast (pysteps + NEXRAD Level 2)

Built by [Shreyas](https://shreyas.uk).
