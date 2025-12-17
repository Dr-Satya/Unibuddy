# Unibuddy

A Python project for the Unibuddy codebase.

## Overview
This repository contains the Unibuddy application source code under the src/ directory. It includes modules for configuration, database access, services, scraping, threaded processing, and application entry points.

## Project Structure
- src/
  - main.py — application entry point
  - config.py — configuration utilities
  - database.py — database utilities
  - services.py — core services
  - scraper.py, 	hreaded_scraper.py — scraping utilities
  - 	hreaded_* — threaded processing helpers
  - models.py — data models
  - performance_monitor.py — performance utilities

## Prerequisites
- Python 3.9+

## Setup
Create and activate a virtual environment (optional but recommended):

`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
`

Install project dependencies if a requirements file is available:

`powershell
pip install -r requirements.txt
`

## Run
If the app is started from src/main.py:

`powershell
python src/main.py
`

Or as a module:

`powershell
python -m src.main
`

## License
Specify your license (e.g., MIT) here.
