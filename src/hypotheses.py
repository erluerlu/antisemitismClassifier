HYPOTHESES = {
    "1": {
        "en": "antisemitic",
        "de": "antisemitisch."
    },
    "0": {
        "en": "not antisemitic",
        "de": "nicht antisemitisch."
    }
}

NLI_LABEL2ID = {"entailment": 2, "neutral": 1, "contradiction": 0}
ID2NLI = {v: k for k, v in NLI_LABEL2ID.items()}