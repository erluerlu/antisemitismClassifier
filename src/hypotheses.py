HYPOTHESES = {
    "1": {
        "en": "antisemitic following IHRA definition",
        "de": "antisemitisch nach IHRA-Definition."
    },
    "0": {
        "en": "not antisemitic following IHRA definition",
        "de": "nicht antisemitisch nach IHRA-Definition."
    }
}

NLI_LABEL2ID = {"entailment": 2, "neutral": 1, "contradiction": 0}
ID2NLI = {v: k for k, v in NLI_LABEL2ID.items()}