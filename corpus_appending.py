import json
from tqdm import tqdm

print("Reading IMDB films")

films = {}
with open("../data.tsv", encoding="utf8") as file:
    for idx, line in enumerate(file):
        if idx == 0:
            continue
        line = line.strip().split("\t")
        films[line[0]] = {
            'title_type': line[1],
            'primaryTitle': line[2],
            'originalTitle': line[3],
            'isAdult': line[4],
            'startYear': line[5],
            'endYear': line[6],
            'runtimeMinutes': line[7],
            'genres': line[8]
        }

print(f"Ended reading the {len(films)} films in IMDB")

# ------------------------------------------------------------------------------------------
print("\n\n")
print("Reading recording new data to films")


with open("../Movies/documents.json", "r+", encoding="utf8") as og_file, open("../concat_films.json", "w+", encoding="utf8") as new_file:
    for line in og_file:
        document = json.loads(line)
    
        doc_id = document['id']
        
        document['title_type'] = films[doc_id]['title_type'] if doc_id in films else ""
        document['startYear'] = films[doc_id]['startYear'] if doc_id in films else ""
        document['genres'] = films[doc_id]['genres'] if doc_id in films else ""

        json.dump(document, new_file)