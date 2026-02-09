HYPOTHESES = {
    "1": {
        "en": "The text contains antisemitism.",
        "de": "Der Text enthält Antisemitismus."
    },
    "0": {
        "en": "The text is not antisemitic.",
        "de": "Der Text ist nicht antisemitisch."
    }
}

NLI_LABEL2ID = {"entailment": 2, "neutral": 1, "contradiction": 0}
ID2NLI = {v: k for k, v in NLI_LABEL2ID.items()}