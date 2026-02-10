import re
import emoji
import pandas as pd
from stopwords import get_stopwords

def clean_text_advanced(csv_path):
    """
    Clean and preprocess text data from a CSV file.
    
    Removes various types of noise and unwanted elements from the 'Text' column
    in the input CSV file, including URLs, emojis, mentions, hashtags, HTML entities,
    and extra whitespace. Outputs the cleaned data to a new CSV file.
    
    Args:
        csv_path (str): Path to the input CSV file containing 'Text' and 'Biased' columns.
    
    Returns:
        None. Writes cleaned data to a new CSV file with '_cleaned' suffix appended
        to the original filename (e.g., 'data.csv' -> 'data_cleaned.csv').
    
    Notes:
        - Rows with missing values in 'Text' or 'Biased' columns are excluded.
        - The following elements are removed from text:
            * URLs (http, https, www)
            * Emojis
            * Mentions (@username)
            * Hashtags (#word)
            * HTML entities (&entity;)
            * Multiple consecutive whitespaces
    """

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Text", "Biased"])

    for index, row in df.iterrows():
        text = row["Text"]

# Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove emojis (more precise)
    text = emoji.replace_emoji(text, replace='')
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (optional)
    text = re.sub(r'#\w+', '', text)
    # Remove HTML entities
    text = re.sub(r'&\w+;', '', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

        
    df.to_csv(f'{csv_path.split(".")[0]}_cleaned.csv', index=False)


def frequencyAnalysis(csv_files=None, top_n=20, language='all', exclude_stopwords=True):
    """
    Analyzes word frequencies in text columns.
    
    Args:
        csv_files (list): List of CSV file paths. Default: ['data/de_cleaned.csv', 'data/en_cleaned.csv']
        top_n (int): Number of most frequent words to display. Default: 20
        language (str): 'de', 'en' or 'all'. Default: 'all'
        exclude_stopwords (bool): Exclude stopwords. Default: True
    
    Returns:
        dict: Frequency analysis per language
    """
    from collections import Counter
    
    if csv_files is None:
        csv_files = {
            'de': 'data/de_cleaned.csv',
            'en': 'data/en_cleaned.csv'
        }
    
    results = {}
    
    # Filter desired languages
    if language != 'all':
        csv_files = {language: csv_files.get(language)}
    
    for lang, filepath in csv_files.items():
        try:
            df = pd.read_csv(filepath)
            
            # Combine all texts
            all_text = ' '.join(df['Text'].dropna().astype(str))
            
            # Convert to lowercase and split into words
            words = re.findall(r'\b\w+\b', all_text.lower())
            
            # Filter stopwords (if desired)
            if exclude_stopwords:
                stopwords = get_stopwords(lang)
                words = [w for w in words if w not in stopwords]
            
            # Count frequencies
            word_freq = Counter(words)
            most_common = word_freq.most_common(top_n)
            
            results[lang] = most_common
            
            print(f"\n{'='*50}")
            print(f"Top {top_n} words in {lang.upper()}: {filepath}")
            if exclude_stopwords:
                print("(Stopwords excluded)")
            print(f"{'='*50}")
            for word, count in most_common:
                print(f"{word:20} : {count:5} mal")
            
        except FileNotFoundError:
            print(f"Warning: File {filepath} not found.")
        except KeyError:
            print(f"Warning: 'Text' column in {filepath} not found.")
    
    return results


def main():
    # Text cleaning
    #csv_path = "data/de.csv"
    #clean_text_advanced(csv_path)
    
    # Word frequency analysis
    frequencyAnalysis(top_n=25)

if __name__ == "__main__":
    main()