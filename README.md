# Document Q&A Agent — Stochastic Inc. Competency Assessment

**One-line:** Lightweight Document Q&A AI Agent (PDF ingestion, RAG, optional ArXiv integration).

## Overview
This repository contains a compact Document Q&A AI Agent built for the Stochastic Inc. Junior AI Engineer competency assessment. It ingests multiple PDFs, extracts text/tables/images, builds a simple vector index using OpenAI embeddings, and serves a small Streamlit UI for querying.

## Features
- Multi-PDF ingestion (text, images, table extraction).
- Chunking + embeddings + retrieval (RAG).
- LLM answer generation (configurable model).
- Optional ArXiv search integration (functionality used when the agent cannot find an answer).
- Streamlit UI for demo and quick testing.

## Quick start
1. Create virtual env and install:
