# Stopwords (filler words) for German and English

GERMAN_STOPWORDS = {
    'und', 'oder', 'aber', 'in', 'auf', 'an', 'zu', 'für', 'von', 'mit', 'durch', 'aus', 'nach',
    'ist', 'sind', 'war', 'waren', 'bin', 'bist', 'seid', 'sein', 'habe', 'habt', 'hast', 'hat',
    'haben', 'hätte', 'hätten', 'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr', 'sie', 'mich', 'dich',
    'ihn', 'uns', 'euch', 'mein', 'dein', 'sein', 'unser', 'euer', 'meine', 'deine', 'seine',
    'unsere', 'eure', 'meinen', 'deinen', 'seinen', 'unseren', 'euren', 'meiner', 'deiner',
    'seiner', 'unserer', 'eurer', 'was', 'welche', 'welcher', 'welches', 'wer', 'wen', 'wem',
    'warum', 'wie', 'wo', 'wann', 'alle', 'jede', 'jeder', 'jedes', 'einige', 'einiger',
    'einige', 'solche', 'solcher', 'solches', 'dasselbe', 'derselbe', 'dieselbe', 'ebenso',
    'ganz', 'ganze', 'ganzer', 'ganzes', 'sehr', 'mehr', 'weniger', 'viel', 'wenig', 'einige',
    'noch', 'nur', 'so', 'ebenso', 'ebenfalls', 'jemals', 'nie', 'nimals', 'immer',
    'eben', 'damals', 'daher', 'danach', 'darunter', 'darüber', 'darauf', 'davor', 'darin',
    'da', 'dort', 'hier', 'hin', 'her', 'häufig', 'oft', 'selten', 'je', 'desto', 'umso',
    'doch', 'deshalb', 'deswegen', 'darum', 'nämlich', 'allerdings', 'jedoch', 'zwar',
    'sondern', 'vielmehr', 'übrigens', 'schliesslich', 'endlich', 'schließlich', 'nicht', 'nein',
    'die', 'der', 'n', 'das', 'nicht'
}

ENGLISH_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'ought',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'what', 'which', 'who', 'whom', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'same', 'so', 'than', 'too', 'very',
    'as', 'am', 'being', 'been', 'be', 'have', 'having', 'does', 'doing', 'did', 'done',
    'having', 'if', 'because', 'while', 'when', 'where', 'after', 'before', 'above', 'below',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'any', 'being', 'these', 'that', 'this',
    'those', 'yourself', 'yourselves', 'ourselves', 'themselves', 'myself', 'himself', 'herself',
    'itself', 'just', 'don', 'should', 'now', 's', 't', 'can', 'will', 'won', 'should',
    'don', 'doesn', 'haven', 'hasn', 'isn', 'aren', 'weren', 'wouldn', 'shouldn', 'couldn',
    'mightn', 'mustn', 'i', 'me', 'we', 'him', 'her', 'them', 'they', 'he', 'she', 'it'
}

STOPWORDS = {
    'de': GERMAN_STOPWORDS,
    'en': ENGLISH_STOPWORDS
}


def get_stopwords(language='all'):
    """
    Returns stopwords for a language.
    
    Args:
        language (str): 'de', 'en' or 'all'. Default: 'all'
    
    Returns:
        set or dict: Stopwords for the selected language(s)
    """
    if language == 'de':
        return GERMAN_STOPWORDS
    elif language == 'en':
        return ENGLISH_STOPWORDS
    elif language == 'all':
        return STOPWORDS
    else:
        raise ValueError(f"Unknown language: {language}")
