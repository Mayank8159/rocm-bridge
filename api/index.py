from http.server import BaseHTTPRequestHandler
import os
import sys

# Directing the request to start the streamlit process
def handler(request):
    os.system("streamlit run app/main.py --server.port 8080")