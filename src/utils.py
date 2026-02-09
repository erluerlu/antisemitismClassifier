import re
import emoji
import pandas as pd

def clean_text_advanced(fileName, columnName):


    text = ''
    # URLs entfernen
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Emojis entfernen (präziser)
    text = emoji.replace_emoji(text, replace='')
    # Mentions entfernen (@username)
    text = re.sub(r'@\w+', '', text)
    # Hashtags entfernen (optional)
    text = re.sub(r'#\w+', '', text)
    # HTML-Entities entfernen
    text = re.sub(r'&\w+;', '', text)
    # Mehrfache Leerzeichen
    text = re.sub(r'\s+', ' ', text).strip()
    return text

#df = pd.read_csv(fileName)
#df['cleaned'] = df[columnName].apply(clean_text_advanced)
#df.to_csv('tweets_cleaned.csv', index=False)