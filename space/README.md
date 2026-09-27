---
title: Customer Review Insights and Response Generator
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
---

Group 5 capstone (TalentSprint, Applied Generative AI and Agentic AI).

Reads product reviews in six languages, finds sentiment, topic and the issue behind a complaint, drafts a reply in the
customer's language, and detects spikes in negative reviews. Replies are drafts; negative reviews always go to a human.

Tabs: saved examples (no cost), your own review (live Claude, spend-capped), spike detection (no cost).

Secrets to set in the Space settings: `ANTHROPIC_API_KEY` (required for the live tab), `DEMO_ACCESS_CODE` (optional),
`MAX_SPEND_USD` (optional, default 0.5).
