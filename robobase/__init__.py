__version__ = "1.0.0"

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(os.path.join(Path(os.path.dirname(__file__)).parent, ".env"), override=True)
