import requests
from bs4 import BeautifulSoup
import csv

URL = input("Enter URL: ")
r = requests.get(URL)
soup = BeautifulSoup(r.content, 'html5lib')

page_text = soup.get_text()

filename = input("Enter the filename for the CSV (without extension): ") + ".csv"

with open(filename, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Page Text"])
    writer.writerow([page_text])

print(f"The content has been saved to {filename}")