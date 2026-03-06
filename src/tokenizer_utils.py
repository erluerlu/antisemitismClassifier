from transformers import AutoTokenizer


def load_tokenizer(tokenizer_name_or_path: str):
    """
    Load a tokenizer with Mistral regex fix enabled when supported.

    Falls back to standard loading for older Transformers versions that do not
    support the ``fix_mistral_regex`` argument.
    """
    try:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path, fix_mistral_regex=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(tokenizer_name_or_path)
