from pathlib import Path

# Diretório raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent

# Diretórios
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Dataset utilizado na reprodução
JDOCTOR_PARAM_10_SR = (
    DATA_DIR
    / "jdoctor"
    / "paramTag_10_SR.json"
)

# Configuração do modelo (utilizada na próxima etapa)

PROVIDER = "ollama"
MODEL_NAME = "Qwen3:8B"
TEMPERATURE = 0
MAX_OUTPUT_TOKENS = 128
