import requests
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_URL = "https://wolnelektury.pl/api/books/"

output_dir = PROJECT_ROOT / "data" / "downloaded" / "books_pl"
output_dir.mkdir(parents=True, exist_ok=True)

response = requests.get(API_URL, timeout=30)
books = response.json()

downloaded = 0

for book in books[:500]:
    try:
        slug = book["slug"]
        title = book["title"]

        details = requests.get(
            f"https://wolnelektury.pl/api/books/{slug}/",
            timeout=30
        ).json()

        txt_url = details.get("txt")

        if not txt_url:
            continue

        text = requests.get(txt_url, timeout=60)

        if text.status_code != 200:
            continue

        filename = output_dir / f"{slug}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(text.text)

        downloaded += 1
        print(f"[{downloaded}] {title}")

    except Exception as e:
        print("Błąd:", e)

print(f"Pobrano {downloaded} książek")