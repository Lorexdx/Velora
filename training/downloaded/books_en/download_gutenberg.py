import requests
from pathlib import Path

BOOKS = {
    "alice_in_wonderland": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "sherlock_holmes": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
}

output_dir = Path("data/downloaded/books_en")
output_dir.mkdir(parents=True, exist_ok=True)

for name, url in BOOKS.items():
    print(f"Pobieranie: {name}")

    response = requests.get(url, timeout=60)

    if response.status_code == 200:
        file_path = output_dir / f"{name}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        print(f"Zapisano: {file_path}")
    else:
        print(f"Błąd: {url}")