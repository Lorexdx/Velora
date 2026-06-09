import requests
from pathlib import Path

BOOKS = {
    "pan_tadeusz": "https://wolnelektury.pl/media/book/txt/pan-tadeusz.txt",
    "lalka": "https://wolnelektury.pl/media/book/txt/lalka-tom-pierwszy.txt",
}

output_dir = Path("data/downloaded/books_pl")
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