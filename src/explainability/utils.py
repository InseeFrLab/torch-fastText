"""
Utility functions.
"""
import ctypes
from typing import Tuple
from scipy.special import softmax
import numpy as np

def get_top_tokens(text, tokenized_text, id_to_token_dicts, attr, top_k=5, padding_index=2009603, end_of_string_index = 0):

    # will contain the list of top tokens (one list per sentence in the text)
    all_top_tokens = [] 

    # iterate over sentences
    for i in range(len(text)):
        id_to_token = id_to_token_dicts[i]
        attr_i = attr[i]
        _, top_k_indices = attr_i.sort()
        assert top_k <= len(top_k_indices), f"Please choose top_k lower than {len(top_k_indices) + 1}."
        top_k_indices = top_k_indices[-top_k:]
        target_token = tokenized_text[i, top_k_indices]
        top_tokens = []
        for ind in target_token:
            if ind.item() != padding_index and ind.item() != end_of_string_index:
                top_tokens.append(id_to_token[ind.item()])
        
        all_top_tokens.append(top_tokens)
    
    return all_top_tokens

def tokenized_text_in_tokens(tokenized_text, id_to_token_dicts, padding_index=2009603, end_of_string_index = 0):
    return [
        [
            id_to_token_dicts[i][token_id.item()] 
            for token_id in tokenized_sentence
            if token_id.item() not in {padding_index, end_of_string_index}
        ]
        for i, tokenized_sentence in enumerate(tokenized_text)
    ]

def match_token_to_word(sentence, list_tokens):
    """
    Match each token to a list of preprocessed words, at sentence level.

    Args:
        sentence (str): Sentence.
        list_tokens (List[str]): List of tokens (in string form).
    
    Returns:
        Dict[str, List[str]]: Mapping from token to list of preprocessed words.
    """

    words = sentence.split()

    # preprocess tokens
    token_to_word = {}
    for token in list_tokens:

        #Preprocess token itself
        preprocessed_token = token.replace('</s>', '')
        preprocessed_token = preprocessed_token.replace('<', '')
        preprocessed_token = preprocessed_token.replace('>', '')

        preprocessed_token = preprocessed_token.split()
        matching_words = []
        for i, tok in enumerate(preprocessed_token):
            # Find all the preprocessed words that contain the token
            matching_word = next((word for word in words if tok in word), None)
            matching_words.append(matching_word)
        token_to_word[token] = matching_words
    return token_to_word


# at text level
def compute_word_score(text, tokenized_text, scores, id_to_token_dicts, token_to_id_dicts, 
                       padding_index=2009603, end_of_string_index=0):

    """
    Compute preprocessed word scores based on token scores.

    Args:
        text (List[str]): List of sentences.
        tokenized_text (List[List[int]]): For each sentence, list of token IDs.
        scores (List[torch.Tensor]): For each sentence, list of token scores.
        id_to_token_dicts (List[Dict[int, str]]): For each sentence, mapping from token ID to token in string form.
        token_to_id_dicts (List[Dict[str, int]]): For each sentence, mapping from token (string) to token ID.
        padding_index (int): Index of padding token.
        end_of_string_index (int): Index of end of string token.

    Returns:
        List[Dict[str, float]]: For each sentence, mapping from preprocessed word to score.
    """
    
    # Convert token IDs to tokens
    tokenized_text_tokens = tokenized_text_in_tokens(tokenized_text, id_to_token_dicts)
    
    word_to_score_dicts = []
    
    for idx, sentence in enumerate(text):
        tokenized_sentence_tokens = tokenized_text_tokens[idx] # sentence level, List[str]

        # Match each token to a list preprocessed words
        token_to_word = match_token_to_word(sentence, tokenized_sentence_tokens) # Dict[str, List[str]]

        id_to_token = id_to_token_dicts[idx] # Dict[int, str]
        score_sentence = scores[idx] # torch.Tensor, token scores
        tokenized_sentence = tokenized_text[idx] # torch.Tensor

        # Initialize word-to-score dictionary with zero values
        word_to_score = {word: 0 for word in sentence.split()}

        # Calculate the score for each token and map to words
        for token_id, score in zip(tokenized_sentence, score_sentence):
            token_id = token_id.item()
            if token_id not in {padding_index, end_of_string_index}:
                token = id_to_token[token_id]
                for word in token_to_word[token]:
                    word_to_score[word] += score.item()

        word_to_score_dicts.append(word_to_score)

    # Apply softmax and format the scores
    for word_to_score in word_to_score_dicts:
        values = np.array(list(word_to_score.values()), dtype=float)
        softmax_values = np.round(softmax(values), 3)
        word_to_score.update({word: float(softmax_value) 
                              for word, softmax_value in zip(word_to_score.keys(), softmax_values)})

    return word_to_score_dicts