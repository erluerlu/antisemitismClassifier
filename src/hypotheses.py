HYPOTHESES = {
    "1": {
        "en": [
            "The text contains antisemitism.",
            "The text dehumanizes, demonizes, or stereotypes Jews as a group.",
            "The text alleges conspiracies or disproportionate power by Jews.",
            "The text denies, trivializes, or distorts the Holocaust.",
            "The text assigns collective blame to Jews or endorses exclusion, discrimination, or violence against them.",
            "The text applies antisemitic delegitimization or double standards toward Israel as collective Jewish self-determination.",
        ],
        "de": [
            "Der Text enthaelt Antisemitismus.",
            "Der Text wertet Juden als Gruppe pauschal ab.",
            "Der Text schreibt Juden kollektiv Schuld, Kontrolle oder Verschwoerungen zu.",
            "Der Text beleidigt oder entmenschlicht Juden als Gruppe.",
            "Der Text fordert Ausgrenzung, Benachteiligung oder Gewalt gegen Juden.",
            "Der Text nutzt antisemitische Chiffren oder Codes gegen Juden.",
            "Der Text stellt antisemitische Feindbilder ueber Juden als berechtigt oder normal dar.",
        ],
    },
    "0": {
        "en": [
            "The text is not antisemitic.",
            "The text discusses Jews, Israel, or politics without degrading Jews as a group.",
            "The text does not dehumanize, demonize, or stereotype Jews.",
            "The text does not assign collective blame or conspiracy claims to Jews.",
            "The text does not deny, trivialize, or distort the Holocaust.",
            "The text does not endorse exclusion, discrimination, or violence against Jews.",
        ],
        "de": [
            "Der Text ist nicht antisemitisch.",
            "Der Text kritisiert Politik oder Regierungshandeln, ohne Juden als Gruppe abzuwerten.",
            "Der Text erwaehnt Juden oder Israel neutral oder berichtend.",
            "Der Text verurteilt Antisemitismus oder schuetzt judische Personen und Gemeinschaften.",
            "Der Text diskutiert den Nahostkonflikt ohne pauschale Schuldzuweisung an Juden.",
            "Der Text zitiert eine problematische Aussage, ohne ihr zuzustimmen.",
            "Der Text enthaelt keine pauschale Abwertung, Entmenschlichung oder Feindbildkonstruktion gegen Juden.",
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