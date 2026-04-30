HYPOTHESES = {
    "1": {
        "de": [
            # 🔴 Direkte Feindseligkeit
            "Der Text enthält Hass, Feindseligkeit oder Abwertung gegenüber Juden.",
            "Der Text stellt Juden als negativ, gefährlich oder minderwertig dar.",

            # 🔴 Verschwörung / Macht-Narrative
            "Der Text behauptet oder suggeriert, dass Juden übermäßigen Einfluss oder Kontrolle über Medien, Politik oder Wirtschaft haben.",
            "Der Text impliziert, dass Juden im Hintergrund die Kontrolle ausüben oder Ereignisse steuern.",

            # 🔴 Kollektivschuld
            "Der Text macht Juden als Gruppe für politische, wirtschaftliche oder historische Ereignisse verantwortlich.",
            "Der Text schreibt Juden pauschal negative Eigenschaften oder Handlungen zu.",

            # 🔴 Israel-/Zionismus-Verschiebung
            "Der Text verwendet Israel oder Zionisten als Ersatz für Juden, um negative Aussagen über Juden zu machen.",
            "Der Text greift Israel oder Zionisten in einer Weise an, die auf Juden als gesamte Gruppe abzielt.",

            # 🔴 „Ich habe nichts gegen Juden, aber…“
            "Der Text relativiert antisemitische Aussagen durch scheinbare Distanzierung von Juden.",
            "Der Text enthält Aussagen wie 'Ich habe nichts gegen Juden, aber' gefolgt von negativen Verallgemeinerungen.",

            # 🔴 Dogwhistles / Codesprache
            "Der Text verwendet stereotype oder codierte Sprache, die typisch für antisemitische Narrative ist.",
            "Der Text deutet antisemitische Inhalte indirekt oder implizit an.",

            # 🔴 Holocaust / NS-Vergleiche
            "Der Text relativiert, leugnet oder verzerrt den Holocaust.",
            "Der Text nutzt unangemessene Nazi-Vergleiche gegen Juden oder Israel.",

            # 🔴 Entmenschlichung
            "Der Text entmenschlicht Juden oder stellt sie als weniger wert dar.",

            # 🔴 Klassische Narrative
            "Der Text behauptet, dass Juden Konflikte oder Kriege absichtlich verursachen.",
            "Der Text stellt Juden als eine koordinierte oder verschwörerische Gruppe dar.",
        ],

        "en": [
            # 🔴 Hostility
            "The text expresses hostility, hatred, or derogatory attitudes toward Jews.",
            "The text portrays Jews as dangerous, inferior, or negative.",

            # 🔴 Conspiracy
            "The text claims or suggests that Jews have excessive influence over media, politics, or finance.",
            "The text implies that Jews secretly control events or institutions.",

            # 🔴 Collective blame
            "The text blames Jews as a group for political, economic, or historical events.",
            "The text assigns negative traits or actions to Jews as a group.",

            # 🔴 Israel substitution
            "The text uses Israel or Zionists as a proxy to make negative claims about Jews.",
            "The text criticizes Israel or Zionists in a way that targets Jews as a whole.",

            # 🔴 Disclaimer pattern
            "The text uses disclaimers like 'I have nothing against Jews, but' followed by generalizations.",
            "The text downplays antisemitism while still expressing harmful stereotypes.",

            # 🔴 Dogwhistles
            "The text uses coded or indirect language associated with antisemitic narratives.",
            "The text implies antisemitic ideas without stating them explicitly.",

            # 🔴 Holocaust distortion
            "The text denies, trivializes, or distorts the Holocaust.",
            "The text makes inappropriate Nazi comparisons involving Jews or Israel.",

            # 🔴 Dehumanization
            "The text dehumanizes Jews or portrays them as less worthy.",

            # 🔴 Classic tropes
            "The text claims that Jews are responsible for causing conflicts or wars.",
            "The text portrays Jews as a coordinated or conspiratorial group.",
        ],
    },

    "0": {
        "de": [
            "Der Text enthält keine Feindseligkeit oder Abwertung gegenüber Juden.",
            "Der Text beschreibt politische oder gesellschaftliche Themen ohne antisemitische Inhalte.",
            "Der Text kritisiert die Politik eines Staates, ohne Juden als Gruppe anzugreifen.",
            "Der Text enthält keine pauschalen Aussagen über Juden.",
            "Der Text verwendet keine antisemitischen Stereotype oder Narrative.",
            "Der Text ist neutral oder sachlich in Bezug auf Juden.",
            "Der Text erwähnt Juden oder Israel ohne negative Verallgemeinerung.",
        ],

        "en": [
            "The text does not express hostility or derogatory attitudes toward Jews.",
            "The text discusses political or social issues without antisemitic content.",
            "The text criticizes a government or policy without targeting Jews as a group.",
            "The text does not make generalizations about Jews.",
            "The text does not use antisemitic stereotypes or narratives.",
            "The text is neutral or factual regarding Jews.",
            "The text mentions Jews or Israel without negative generalization.",
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