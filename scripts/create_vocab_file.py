import argparse
import ast
import unicodedata
import json


def is_diacritic(char):
    """Check if a character is a diacritic."""
    if len(char) != 1:  # Ensure it's a single character
        return False
    return unicodedata.category(char) in ['Mn', 'Mc', 'Me']


def clean_vocab(train_dev_set, allowed_punctuations):
    """
    Clean the train_dev_set by removing diacritics, duplicates, and unnecessary punctuations.
    
    Args:
        train_dev_set (set): Set of characters.
        allowed_punctuations (set): Punctuation characters to retain.
    
    Returns:
        list: Cleaned and sorted list of unique characters.
    """
    cleaned_vocab = set()
    for char in train_dev_set:
        if not isinstance(char, str) or len(char) != 1:
            continue  # Skip invalid entries
        if not is_diacritic(char):
            # Check if character is punctuation
            if unicodedata.category(char).startswith('P'):
                if char in allowed_punctuations:
                    cleaned_vocab.add(char)
            else:
                cleaned_vocab.add(char)
    # Sort the vocab for consistency
    return sorted(cleaned_vocab)


def write_vocab(vocab, output_file):
    """
    Write the vocabulary to a file, one token per line.
    
    Args:
        vocab (list): List of vocabulary tokens.
        output_file (str): Path to the output vocab file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(token + '\n')
    print(f"Vocabulary successfully written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create a clean vocabulary file from train_dev_set.')
    parser.add_argument('--train_dev_set', type=str, required=True,
                        help="Pass the train_dev_set variable either as a JSON file or an inline Python list.")
    parser.add_argument('--output_vocab', type=str, default='vocab.txt', help='Path to save the vocab file.')
    parser.add_argument('--punctuations', type=str, default="[' ', '۔']", 
                        help='Comma-separated punctuations to retain (e.g., " , ۔").')
    args = parser.parse_args()

    # Load the train_dev_set
    try:
        if args.train_dev_set.endswith('.json'):  # Assume JSON file input
            with open(args.train_dev_set, 'r', encoding='utf-8') as f:
                train_dev_set = set(json.load(f))
        else:  # Assume inline Python list
            train_dev_set = set(ast.literal_eval(args.train_dev_set))
    except Exception as e:
        raise ValueError(f"Invalid train_dev_set input: {e}")

    # Parse allowed punctuations
    try:
        allowed_punctuations = set(ast.literal_eval(args.punctuations))
    except Exception as e:
        raise ValueError(f"Invalid punctuations input: {e}")

    # Clean the vocab
    cleaned_vocab = clean_vocab(train_dev_set, allowed_punctuations)

    # Write the vocab to a file
    write_vocab(cleaned_vocab, args.output_vocab)


if __name__ == '__main__':
    main()

