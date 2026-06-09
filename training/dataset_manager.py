from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DatasetManager:
    def __init__(self):
        self.downloaded_dir = PROJECT_ROOT / "data" / "downloaded"
        self.final_dir = PROJECT_ROOT / "data" / "final"

    def list_datasets(self):
        return [file.name for file in self.downloaded_dir.iterdir() if file.is_file()]

    def create_directories(self):
        self.downloaded_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)