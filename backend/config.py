from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MAX_PROMPT_LENGTH = 500
DEFAULT_MODEL_NAME = "shap-e"
