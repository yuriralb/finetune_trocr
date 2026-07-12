"""Gera metadata.csv com caminhos relativos e quoting correto."""
import csv

with open("metadata.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["file_name", "text"])

    c = 1
    with open("strings.txt", "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                writer.writerow([f"frases/image{c}.JPEG", line])
                c += 1
