"""
Utility functions.
"""
import ctypes
from typing import Tuple
from scipy.special import softmax
import numpy as np
import difflib
from collections import Counter

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

def preprocess_token(token):
    preprocessed_token = token.replace('</s>', '')
    preprocessed_token = preprocessed_token.replace('<', '')
    preprocessed_token = preprocessed_token.replace('>', '')

    preprocessed_token = preprocessed_token.split()

    return preprocessed_token


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
        preprocessed_token = preprocess_token(token)

        matching_words = []
        for i, tok in enumerate(preprocessed_token):
            
            # Find all the preprocessed words that contain the token
            for word in words:
                if tok in word:
                    matching_words.append(word)

        token_to_word[token] = matching_words
    return token_to_word

def levenshtein(s1, s2):
    """
    Calculate the Levenshtein distance between two strings.
    Returns a normalized score between 0 and 1, where 1 means identical.
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return 0.0

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    # Normalize by the length of the longer string
    max_len = max(len(s1), len(s2))
    return 1 - (previous_row[-1] / max_len)

def jaro_winkler(s1, s2, p=0.1):
    """
    Calculate Jaro-Winkler similarity between two strings.
    p is the scaling factor for how much to favor matching prefixes.
    """
    # If strings are equal
    if s1 == s2:
        return 1.0

    # Find length of strings
    len1 = len(s1)
    len2 = len(s2)

    # Maximum distance between two chars to be considered matching
    match_distance = (max(len1, len2) // 2) - 1

    # Arrays of booleans that indicate matching/non-matching chars
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    # Number of matches and transpositions
    matches = 0
    transpositions = 0

    # Find matching characters within match_distance
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    # Jaro similarity
    transpositions = transpositions // 2
    jaro = ((matches / len1) + (matches / len2) + 
            ((matches - transpositions) / matches)) / 3.0

    # Find length of common prefix (up to 4 chars)
    prefix_len = 0
    max_prefix_len = min(4, min(len1, len2))
    while prefix_len < max_prefix_len and s1[prefix_len] == s2[prefix_len]:
        prefix_len += 1

    # Jaro-Winkler similarity
    return jaro + (prefix_len * p * (1 - jaro))

def longest_common_subsequence(s1, s2):
    """
    Calculate length of longest common subsequence between two strings.
    Returns a normalized score between 0 and 1.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Normalize by the length of the shorter string
    return dp[m][n] / min(m, n)

def ngram_cosine_similarity(s1, s2, n=2):
    """
    Calculate cosine similarity between character n-grams of two strings.
    """
    # Generate n-grams
    def get_ngrams(s, n):
        return [s[i:i+n] for i in range(max(0, len(s)-n+1))]
    
    # Get n-gram counts
    vec1 = Counter(get_ngrams(s1.lower(), n))
    vec2 = Counter(get_ngrams(s2.lower(), n))
    
    # Find common n-grams
    intersection = set(vec1.keys()) & set(vec2.keys())
    
    # Calculate cosine similarity
    numerator = sum(vec1[x] * vec2[x] for x in intersection)
    sum1 = sum(val*val for val in vec1.values())
    sum2 = sum(val*val for val in vec2.values())
    denominator = np.sqrt(sum1) * np.sqrt(sum2)
    
    if not denominator:
        return 0.0
    return numerator / denominator

def map_processed_to_original(processed_words, original_words , n=5, cutoff=0):
    """
    Find best matches using multiple similarity metrics.
    """
    matches = {}
    
    for orig in original_words:
        similarities = []
        for proc in processed_words:
            # Calculate different similarity scores
            lev_score = levenshtein(orig.lower(), proc.lower())
            jw_score = jaro_winkler(orig.lower(), proc.lower())
            lcs_score = longest_common_subsequence(orig.lower(), proc.lower())
            cos_score = ngram_cosine_similarity(orig, proc)
            
            # Combine scores (you can adjust weights)
            final_score = max(
                lev_score * 0.2,
                jw_score * 0.8,  # Give more weight to Jaro-Winkler
                lcs_score * 0,
                cos_score * 0.2
            )
            
            if final_score >= cutoff:
                similarities.append((proc, final_score))
        
        # Sort by score and take top n
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_matches = similarities[:n]
        # Split into separate lists for words and scores
        matched_words, scores = zip(*top_matches) if top_matches else ([], [])
        matches[orig] = (list(matched_words), list(scores))
    
    return matches

# def map_processed_to_original(processed_words, original_words, n=1, cutoff=0.7):
#     """
#     Map processed words to original words based on similarity scores.

#     Args:
#         processed_words (List[str]): List of processed words.
#         original_words (List[str]): List of original words.
#         n (int): Number of closest processed words to consider for a given original word.
#         cutoff (float): Minimum similarity score for a match.
    
#     Returns:
#         Dict[str, Tuple[List[str], List[float]]]: Mapping from original word to tuple of closest processed words and similarity scores.
#     """

#     # For each word in the original list, find the n closest matching processed words
#     word_mapping = {}

#     for original_word in original_words:
#         processed_word_scores = []

#         # Calculate the similarity score for each processed word with the current original word
#         for processed_word in processed_words:
#             similarity_score = difflib.SequenceMatcher(None, processed_word, original_word).ratio() # Ratcliff-Obershelp algorithm

#             # Only consider matches with similarity above the cutoff
#             if similarity_score >= cutoff:
#                 processed_word_scores.append((processed_word, similarity_score))
        
#         # Sort processed words by similarity score in descending order
#         processed_word_scores.sort(key=lambda x: x[1], reverse=True)

#         # Extract the n closest processed words and their similarity scores
#         closest_words = [item[0] for item in processed_word_scores[:n]]
#         similarity_scores = [item[1] for item in processed_word_scores[:n]]

#         # Add the tuple (list of closest words, list of similarity scores) to the mapping
#         word_mapping[original_word] = (closest_words, similarity_scores)

#     return word_mapping


# at text level
def compute_preprocessed_word_score(preprocessed_text, tokenized_text, scores, id_to_token_dicts, token_to_id_dicts, 
                        padding_index=2009603, end_of_string_index=0, aggregate=True):

    """
    Compute preprocessed word scores based on token scores.

    Args:
        preprocessed_text (List[str]): List of preprocessed sentences.
        tokenized_text (List[List[int]]): For each sentence, list of token IDs.
        scores (List[torch.Tensor]): For each sentence, list of token scores.
        id_to_token_dicts (List[Dict[int, str]]): For each sentence, mapping from token ID to token in string form.
        token_to_id_dicts (List[Dict[str, int]]): For each sentence, mapping from token (string) to token ID.
        padding_index (int): Index of padding token.
        end_of_string_index (int): Index of end of string token.
        aggregate (bool): Whether to aggregate scores at word level (if False, stay at token level).

    Returns:
        List[Dict[str, float]]: For each sentence, mapping from preprocessed word to score.
    """
    
    # Convert token IDs to tokens
    tokenized_text_tokens = tokenized_text_in_tokens(tokenized_text, id_to_token_dicts)

    if not aggregate:
        return tokenized_text_tokens
    
    word_to_score_dicts = []
    
    for idx, sentence in enumerate(preprocessed_text):
        tokenized_sentence_tokens = tokenized_text_tokens[idx]  # sentence level, List[str]

        # Match each token to a list preprocessed words
        token_to_word = match_token_to_word(sentence, tokenized_sentence_tokens) # Dict[str, List[str]]

        id_to_token = id_to_token_dicts[idx]  # Dict[int, str]
        score_sentence_topk = scores[idx]  # torch.Tensor, token scores, (top_k, seq_len)
        tokenized_sentence = tokenized_text[idx]  # torch.Tensor


        # Calculate the score for each token and map to words
        word_to_score_topk = []
        for k in range(len(score_sentence_topk)):

            # Initialize word-to-score dictionary with zero values
            word_to_score = {word: 0 for word in sentence.split()}

            score_sentence = score_sentence_topk[k]
            for token_id, score in zip(tokenized_sentence, score_sentence):
                token_id = token_id.item()
                if token_id not in {padding_index, end_of_string_index}:
                    token = id_to_token[token_id]
                    for word in token_to_word[token]:
                        word_to_score[word] += score.item()
            word_to_score_topk.append(word_to_score.copy())

        word_to_score_dicts.append(word_to_score_topk)
    
    # Apply softmax and format the scores
    # for word_to_score_topk in word_to_score_dicts:
    #     for word_to_score in word_to_score_topk:
    #         values = np.array(list(word_to_score.values()), dtype=float)
    #         softmax_values = np.round(softmax(values), 3)
    #         word_to_score.update({word: float(softmax_value) 
    #                              for word, softmax_value in zip(word_to_score.keys(),
    #                              softmax_values)})

    return word_to_score_dicts

def compute_word_score(word_to_score_dicts, text,  n=5, cutoff=0.75):
    """
    Compute word scores based on preprocessed word scores.

    Args:
        word_to_score_dicts (List[List[Dict[str, float]]]): For each sentence, list of top_k mappings from preprocessed word to score.
        text (List[str]): List of sentences.
        n (int): Number of closest preprocessed words to consider for a given original word.
        cutoff (float): Minimum similarity score for a match.
    
    Returns:
        List[List[List[float]]]: For each sentence, list of top-k scores for each word.
    """

    full_all_scores = []
    for idx, word_to_score_topk in enumerate(word_to_score_dicts): # iteration over sentences
        all_scores_topk = [] 
        for word_to_score in word_to_score_topk: # iteration over top_k (the preds)
            processed_words = list(word_to_score.keys())
            original_words = text[idx].split()        

            for i, word in enumerate(original_words):
                original_words[i] = word.replace(',', '')
      
            mapping = map_processed_to_original(processed_words, original_words, n=n, cutoff=cutoff) # Dict[str, Tuple[List[str], List[float]]]
            print(mapping)
            scores = []
            for word in original_words:
                processed_words, distances = mapping[word]
                word_score = 0
                for i, potential_processed_word in enumerate(processed_words):
                    score = word_to_score[potential_processed_word]
                    word_score += score * distances[i] / np.sum(distances) # weighted average (weights = similarity scores)

                scores.append(word_score)

            scores = softmax(scores) # softmax normalization. Length = len(original_words)

            all_scores_topk.append(scores) # length top_k

        full_all_scores.append(all_scores_topk) # length = len(text)

    return full_all_scores
