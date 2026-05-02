HYPOTHESES = {
        "1": {
            "de": [
            "Der Text beleidigt, entmenschlicht oder stereotypisiert Juden.",
            "Der Text verbreitet antisemitische Verschwoerungsnarrative oder behauptet, Juden kontrollieren Politik, Medien oder Geld.",
            "Der Text macht Juden pauschal fuer gesellschaftliche Probleme verantwortlich oder spricht ihnen gleiche Rechte ab.",
            "Der Text setzt Israel mit dem Nationalsozialismus gleich oder spricht Israel pauschal das Existenzrecht ab.",
            "Der Text fordert Gewalt, Ausgrenzung oder Vertreibung von Juden.",
            "Der Text leugnet oder verharmlost den Holocaust oder verdreht historische Fakten, um Juden zu diffamieren."
            ],
            "en": [
            "The text dehumanizes, stereotypes, or insults Jews.",
            "The text spreads antisemitic conspiracy narratives or claims Jews control politics, media, or money.",
            "The text holds Jews collectively responsible for societal problems or denies them equal rights.",
            "The text equates Israel with Nazi Germany or denies Israel's right to exist as a Jewish state.",
            "The text calls for violence, exclusion, or expulsion of Jews.",
            "The text denies or trivializes the Holocaust or distorts historical facts to defame Jews."
            ]
        },
        "0": {
            "de": [
            "Der Text ist eine sachliche oder respektvolle Aussage ohne Feindseligkeit gegen Juden.",
            "Der Text kritisiert eine Regierung oder Politik ohne Juden als Gruppe anzugreifen.",
            "Der Text diskutiert juedisches Leben, Geschichte oder Religion in respektvoller Weise."
            ],
            "en": [
            "The text is a factual or respectful statement without hostility toward Jews.",
            "The text criticizes a government or policy without attacking Jews as a group.",
            "The text discusses Jewish life, history, or religion in a respectful manner."
            ]
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