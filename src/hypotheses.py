HYPOTHESES = {
    "1": {
        "de": [
            # 🔴 Direkte Feindseligkeit & Dämonisierung
            "Der Text äußert Hass, Verachtung oder Feindseligkeit gegenüber Juden als religiöse oder ethnische Gruppe.",
            "Der Text nutzt dehumanisierende Begriffe wie Tiere, Parasiten oder Teufel für jüdische Menschen oder den Staat Israel.",
            
            # 🔴 Verschwörung & Macht
            "Der Text behauptet, dass Juden heimlich die Weltpolitik, das Finanzsystem oder die Medien kontrollieren.",
            "Der Text unterstellt Juden eine angeborene Gier oder eine illegitime Macht durch Geld und Bankwesen.",
            
            # 🔴 Israel-/Zionismus-Verschiebung (Recall-Booster)
            "Der Text setzt das Handeln des Staates Israel oder von Zionisten mit den Verbrechen der Nationalsozialisten gleich.",
            "Der Text verwendet 'Zionisten' als Codewort, um bösartige Eigenschaften auf Juden zu projizieren.",
            "Der Text spricht Israel das Existenzrecht ab oder fordert Gewalt gegen dessen jüdische Bevölkerung.",
            
            # 🔴 Kollektivschuld & Doppelstandards
            "Der Text macht alle Juden weltweit für die Handlungen des Staates Israel oder für historische Ereignisse verantwortlich.",
            "Der Text fordert von Juden oder Israel ein Verhalten, das von keiner anderen demokratischen Nation verlangt wird.",
            
            # 🔴 Holocaust
            "Der Text leugnet, verharmlost oder verzerrt die historischen Fakten des Holocaust.",
        ],
        "en": [
            # 🔴 Hostility & Dehumanization
            "The text expresses hatred, contempt, or hostility toward Jews as a religious or ethnic group.",
            "The text uses dehumanizing terms like animals, parasites, or devils for Jewish people or the State of Israel.",
            
            # 🔴 Conspiracy & Power
            "The text claims that Jews secretly control global politics, the financial system, or the media.",
            "The text implies innate greed or illegitimate power through money and banking.",
            
            # 🔴 Israel/Zionism Shift
            "The text equates the actions of the State of Israel or Zionists with the crimes of the Nazis.",
            "The text uses 'Zionists' as a code word to project malicious traits onto Jews.",
            "The text denies Israel's right to exist or calls for violence against its Jewish population.",
            
            # 🔴 Collective Blame & Double Standards
            "The text holds all Jews worldwide responsible for the actions of the State of Israel or historical events.",
            "The text demands standards of behavior from Jews or Israel not expected of any other democratic nation.",
            
            # 🔴 Holocaust
            "The text denies, trivializes, or distorts the historical facts of the Holocaust.",
        ],
    },
    "0": {
        "de": [
            # 🟢 Sachliche politische Kritik ohne Kollektivzuschreibung
            "Der Text kritisiert staatliches, militärisches oder parteipolitisches Handeln, ohne Juden als Gruppe verantwortlich zu machen.",
            "Der Text beschreibt politische Ereignisse oder Konflikte sachlich, ohne pauschale Urteile über Juden als religiöse oder ethnische Gruppe.",
            "Der Text diskutiert Israel, eine Regierung, Institutionen oder politische Akteure, ohne daraus Aussagen über Juden im Allgemeinen abzuleiten.",

            # 🟢 Differenzierung statt Generalisierung
            "Der Text unterscheidet zwischen einer Regierung, einem Staat, Zionismus oder einzelnen Akteuren und jüdischer Identität als Gruppe.",
            "Der Text enthält keine pauschale negative Zuschreibung gegenüber Juden als religiöse oder ethnische Gemeinschaft.",
            "Der Text verhandelt Politik, Geschichte oder Menschenrechte, ohne Juden kollektiv Motive, Schuld oder Macht zuzuschreiben.",
        ],
        "en": [
            # 🟢 Political criticism without collective blame
            "The text criticizes state, military, or party-political actions without holding Jews as a group responsible.",
            "The text describes political events or conflict in factual terms without sweeping claims about Jews as a religious or ethnic group.",
            "The text discusses Israel, a government, institutions, or political actors without turning this into claims about Jews in general.",

            # 🟢 Differentiation instead of generalization
            "The text distinguishes between a government, a state, Zionism, or individual actors and Jewish identity as a group.",
            "The text contains no blanket negative attribution toward Jews as a religious or ethnic community.",
            "The text addresses politics, history, or human rights without assigning collective motives, guilt, or power to Jews.",
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