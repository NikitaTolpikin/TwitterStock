import csv
import re


def preprocess_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'ССЫЛКА', text)
    text = re.sub('@[^\s]+', 'ПОЛЬЗОВАТЕЛЬ', text)
    text = re.sub('#[^\s]+', 'ХЭШТЕГ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('rt', ' ', text)
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
    text = re.sub('[\s][a-zA-Zа-яА-Я][\s]|^[a-zA-Zа-яА-Я][\s]', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


all_raw = []
marked_raw = []
all_prep = []
marked_prep = []

with open('all.csv', mode='r', encoding='utf-8-sig') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        all_raw.append({'id': row['id'], 'text': row['text']})

with open('marked.csv', mode='r', encoding='utf-8-sig') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        marked_raw.append({'id': row['id'], 'text': row['text'], 'mark': row['mark']})

for data in all_raw:
    all_prep.append({'id': data['id'], 'text': preprocess_text(data['text'])})

i = 0
for data in marked_raw:
    prep_text = preprocess_text(data['text'])
    if prep_text != "ССЫЛКА":
        marked_prep.append({'id': i, 'text': prep_text, 'mark': data['mark']})
        i += 1

with open('all_prep.csv', 'w', encoding='UTF-8') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'text'])
    w.writeheader()
    for data in all_prep:
        w.writerow(data)

with open('marked_prep.csv', 'w', encoding='UTF-8') as f:
    w = csv.DictWriter(f, fieldnames=['id', 'text', 'mark'])
    w.writeheader()
    for data in marked_prep:
            w.writerow(data)


