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
            "Der Text entmenschlicht, daemonisiert oder stereotypisiert Juden als Gruppe.",
            "Der Text behauptet antisemitische Verschwoerungen oder uebermaessige Macht von Juden.",
            "Der Text leugnet, verharmlost oder verzerrt den Holocaust.",
            "Der Text schreibt Juden Kollektivschuld zu oder befuerwortet Ausgrenzung, Diskriminierung oder Gewalt gegen sie.",
            "Der Text delegitimiert Israel antisemitisch oder legt doppelte Standards gegen juedische Selbstbestimmung an.",
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
            "Der Text thematisiert Juden, Israel oder Politik, ohne Juden als Gruppe abzuwerten.",
            "Der Text entmenschlicht, daemonisiert oder stereotypisiert Juden nicht.",
            "Der Text schreibt Juden keine Kollektivschuld oder Verschwoerung zu.",
            "Der Text leugnet, verharmlost oder verzerrt den Holocaust nicht.",
            "Der Text befuerwortet keine Ausgrenzung, Diskriminierung oder Gewalt gegen Juden.",
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