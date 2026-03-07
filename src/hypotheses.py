HYPOTHESES = {
    "1": {
        "en": [
            "The text contains antisemitism.",
            "The text dehumanizes, stereotypes, or insults Jews.",
            "The text promotes hostility toward Jews or Jewish people.",
            "The text normalizes antisemitic conspiracy narratives.",
            "The text questions the right of Israel to exist.",
        ],
        "de": [
            "Der Text enthaelt Antisemitismus.",
            "Der Text beleidigt, entmenschlicht oder stereotypisiert Juden.",
            "Der Text richtet Feindseligkeit gegen Juden oder juedische Menschen.",
            "Der Text normalisiert antisemitische Verschwoerungsnarrative.",
            "Der Text macht Juden pauschal fuer gesellschaftliche Probleme verantwortlich.",
            "Der Text reproduziert antisemitische Chiffren oder Codesprache.",
            "Der Text delegitimiert Juden als Gruppe oder spricht ihnen gleiche Rechte ab.",
            "Der Text verbindet Juden kollektiv mit Schuld, Verrat oder Kontrolle.",
            "Der Text fordert Ausgrenzung, Gewalt oder Bestrafung gegen Juden.",
            "Der Text stellt antisemitische Feindbilder als normal oder berechtigt dar.",
        ],
    },
    "0": {
        "en": [
            "The text is not antisemitic.",
            "The text does not express hostility toward Jews.",
            "The text does not dehumanize or insult Jews.",
            "The text does not question the right of Israel to exist",
        ],
        "de": [
            "Der Text ist nicht antisemitisch.",
            "Der Text richtet keine Feindseligkeit gegen Juden.",
            "Der Text beleidigt oder entmenschlicht Juden nicht.",
            "Der Text kritisiert Politik, ohne Juden als Gruppe abzuwerten.",
            "Der Text enthaelt keine antisemitische Generalisierung ueber Juden.",
        ],
    },
}


def hypotheses_for_class(cls: str, lang: str) -> list[str]:
    """Returns one or multiple hypothesis templates for a class/language."""
    value = HYPOTHESES[cls][lang]
    if isinstance(value, str):
        return [value]
    return list(value)

NLI_LABEL2ID = {"entailment": 2, "neutral": 1, "contradiction": 0}
ID2NLI = {v: k for k, v in NLI_LABEL2ID.items()}