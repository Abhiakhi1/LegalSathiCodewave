# src/knowledge/legal_rules.py
# Lightweight, extensible concept map. Add/adjust as you see new queries.

CONCEPTS = {
    # --- Crimes against person ---
    "assault": {
        "synonyms": [
            "attack", "attacked", "hit me", "beaten", "beat me", "slapped",
            "fight", "threaten", "threatened", "criminal force"
        ],
        "acts": ["Indian Penal Code, 1860"],
        "sections": ["Section 351", "Section 352", "Section 354", "Section 356"],
        "notes": "351 definition of assault; 352 punishment; 354 outraging modesty; 356 assault to commit theft",
    },
    "murder/homicide": {
        "synonyms": ["murder", "murdered", "killed", "homicide", "manslaughter"],
        "acts": ["Indian Penal Code, 1860"],
        "sections": ["Section 299", "Section 300", "Section 302", "Section 304"],
        "notes": "299 culpable homicide; 300 murder; 302 punishment; 304 culpable homicide not amounting to murder",
    },
    "wrongful restraint/confinement": {
        "synonyms": ["stop me to move", "blocked my way", "locked me", "confined me", "kidnapped me"],
        "acts": ["Indian Penal Code, 1860"],
        "sections": ["Section 339", "Section 340", "Section 341", "Section 342"],
        "notes": "339/341 restraint; 340/342 confinement",
    },

    # --- Property / economic offences ---
    "theft": {
        "synonyms": ["stole", "stolen", "robbed", "theft", "pickpocket"],
        "acts": ["Indian Penal Code, 1860"],
        "sections": ["Section 378", "Section 379", "Section 380", "Section 390", "Section 392"],
        "notes": "378 def; 379 punishment; 380 theft in dwelling; 390/392 robbery",
    },
    "cheating": {
        "synonyms": ["fraud", "cheated", "scam", "duped", "bhool bhulaiya", "embezzlement"],
        "acts": ["Indian Penal Code, 1860"],
        "sections": ["Section 415", "Section 417", "Section 420", "Section 406", "Section 409"],
        "notes": "415 cheating; 417 punishment; 420 cheating & dishonestly inducing delivery; 406/409 criminal breach of trust",
    },
    "encroachment/trespass": {
        "synonyms": ["encroachment", "trespass", "boundary crossed", "illegal possession", "land grabbed"],
        "acts": ["Indian Penal Code, 1860"],
        "sections": ["Section 441", "Section 447"],
        "notes": "441 criminal trespass; 447 punishment",
    },

    # --- Family / social offences ---
    "dowry harassment": {
        "synonyms": ["dowry", "harassment for dowry", "dowry death", "498a", "dowry case"],
        "acts": ["Indian Penal Code, 1860", "Dowry Prohibition Act, 1961"],
        "sections": ["Section 498A", "Section 304B"],
        "notes": "498A cruelty by husband/relatives; 304B dowry death",
    },
    "domestic violence": {
        "synonyms": ["domestic violence", "husband beat", "abuse at home", "DV Act"],
        "acts": ["Protection of Women from Domestic Violence Act, 2005"],
        "sections": ["Section 12", "Section 18", "Section 20", "Section 21", "Section 22"],
        "notes": "applications & reliefs under PWDVA",
    },

    # --- Sexual offences ---
    "rape/sexual offence": {
        "synonyms": ["rape", "sexual assault", "sexual violence", "outraged modesty", "molestation"],
        "acts": ["Indian Penal Code, 1860", "POCSO Act, 2012"],
        "sections": ["Section 375", "Section 376", "Section 354", "POCSO Sections (child)"],
        "notes": "375/376 rape; 354 outraging modesty; POCSO for minors",
    },

    # --- Tech ---
    "cyber fraud": {
        "synonyms": ["online fraud", "upi scam", "otp fraud", "phishing", "hacked", "instagram hacked", "cybercrime"],
        "acts": ["Information Technology Act, 2000"],
        "sections": ["Section 66", "Section 66C", "Section 66D", "Section 67"],
        "notes": "identity theft, cheating by personation, publishing obscene material",
    },
}
