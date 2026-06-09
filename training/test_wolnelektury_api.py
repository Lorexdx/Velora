import requests

url = "https://wolnelektury.pl/api/books/"

response = requests.get(url, timeout=30)

print("Status:", response.status_code)

books = response.json()

print("Liczba książek:", len(books))

for book in books[:10]:
    print(book["title"])