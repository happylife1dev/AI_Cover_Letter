cd app
uvicorn main:app --log-level=debug --reload --reload-include *.html
