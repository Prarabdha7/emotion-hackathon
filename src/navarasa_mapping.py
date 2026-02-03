# src/navarasa_mapping.py

NAVARASA_MAP = {
    "Love": "Shringara",
    "Longing": "Shringara",
    "Happiness": "Hasya",
    "Sadness": "Karuna",
    "Melancholy": "Karuna",
    "Anger": "Raudra",
    "Fear": "Bhayanaka",
    "Disgust": "Bibhatsa",
    "Surprise": "Adbhuta",
    "Peace": "Shanta",
    "Devotion": "Shanta / Bhakti",
    "Reverence": "Shanta / Bhakti",
    "Philosophy": "Shanta",
    "Duty": "Veera",
    "Courage": "Veera"
}

def map_to_navarasa(emotion):
    return NAVARASA_MAP.get(emotion, "Shanta")