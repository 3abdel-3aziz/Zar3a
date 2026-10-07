import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "back_end"))
sys.path.insert(0, os.path.join(repo_root, "back_end", "src"))
sys.path.insert(0, os.path.join(repo_root, "src"))

import uvicorn
from api.main import app

if __name__ == "__main__":
    uvicorn.run("run_server:app", host="127.0.0.1", port=8000, reload=True, log_level="info")
