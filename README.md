# Sales Insight Agent

This project provides a natural-language sales analytics assistant using:
- Sales API  
- LLM (Google Gemini 2.5 Flash)  
- CLI interface  

## Setup

```bash
python -m venv venv
source venv/Scripts/activate    # Windows Git Bash
pip install -r requirements.txt
```

## Environment Variable

Generate a Google Gemini API key and add it to your `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

## Run the Project

Your project is ready.  
Give your prompts and the AI Sales Agent will reply with insights.

```
python run.py
```
