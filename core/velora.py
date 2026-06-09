from config.settings import Settings

class Velora:
    def __init__(self):
        self.name = Settings.AI_NAME
        self.version = Settings.VERSION

    def start(self):
        print(f"{self.name} v{self.version} uruchomiona")