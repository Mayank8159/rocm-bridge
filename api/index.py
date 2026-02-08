import os
from flask import Flask # or FastAPI

app = Flask(__name__)

if __name__ == "__main__":
    # Render provides the port via the PORT env var
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)