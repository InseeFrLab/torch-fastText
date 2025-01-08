"""
Utility functions.
"""
import warnings
import difflib
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

from ..preprocess import clean_text_feature


def tokenized_text_in_tokens(
    tokenized_text, id_to_token_dicts, padding_index=2009603, end_of_string_index=0
):
    return [
        [
            id_to_token_dicts[i][token_id.item()]
            for token_id in tokenized_sentence
            if token_id.item() not in {padding_index}
        ]
        for i, tokenized_sentence in enumerate(tokenized_text)
    ]


def preprocess_token(token):
    preprocessed_token = token.replace("</s>", "")
    preprocessed_token = preprocessed_token.replace("<", "")
    preprocessed_token = preprocessed_token.replace(">", "")

    preprocessed_token = preprocessed_token.split()

    return preprocessed_token


def map_processed_to_original(processed_words, original_words, n=1, cutoff=0.9):
    """
    Map processed words to original words based on similarity scores.

    Args:
        processed_words (List[str]): List of processed words.
        original_words (List[str]): List of original words.
        n (int): Number of closest processed words to consider for a given original word.
        cutoff (float): Minimum similarity score for a match.

    Returns:
        Dict[str, str]: Mapping from original word to the corresponding closest processed word.
    """

    # For each word in the original list, find the n closest matching processed words
    word_mapping = {}

    for original_word in original_words:
        original_word_prepro = clean_text_feature([original_word], remove_stop_words=False)[
            0
        ]  # Preprocess the original word

        if original_word_prepro == "":
            continue

        max_similarity_score = 0
        best_processed_word = None
        # Calculate the similarity score for each processed word with the current original word
        for processed_word in processed_words:
            similarity_score = difflib.SequenceMatcher(
                None, processed_word, original_word_prepro
            ).ratio()  # Ratcliff-Obershelp algorithm

            # Only consider matches with similarity above the cutoff
            if similarity_score > max_similarity_score and similarity_score >= cutoff:
                max_similarity_score = similarity_score
                best_processed_word = processed_word

        if best_processed_word is not None:
            # original_word = original_word.replace(',', '')
            # Add the tuple (list of closest words, list of similarity scores) to the mapping
            word_mapping[original_word] = best_processed_word

    return word_mapping


def test_end_of_word(all_processed_words, word, target_token, next_token, min_n):
    flag = False
    if target_token[-1] == ">":
        if next_token[0] == "<":
            if word in target_token:
                flag = True
            if word in next_token:
                flag = False
            if next_token[1] != word[0]:
                flag = True
            if len(next_token) == min_n:
                flag = True
        if next_token in all_processed_words:
            flag = True

    return flag


def match_word_to_token_indexes(sentence, tokenized_sentence_tokens, min_n):
    """
    Match words to token indexes in a sentence.

    Args:
        sentence (str): Preprocessed sentence.
        tokenized_sentence_tokens (List[str]): List of tokenized sentence tokens.

    Returns:
        Dict[str, List[int]]: Mapping from word to list of token indexes.

    """

    pointer_token = 0
    res = {}
    processed_sentence = clean_text_feature([sentence], remove_stop_words=False)[0]
    processed_words = processed_sentence.split()
    # we know the tokens are in the right order
    for index_word, word in enumerate(processed_words):
        if word not in res:
            res[word] = []

        start = pointer_token

        # while we don't reach the end of the word, get going
        while not test_end_of_word(
            processed_words,
            word,
            tokenized_sentence_tokens[pointer_token],
            tokenized_sentence_tokens[pointer_token + 1],
            min_n=min_n
        ):
            pointer_token += 1
            if pointer_token == len(tokenized_sentence_tokens)-1:
                warnings.warn("Error in the tokenization of the sentence")
                # workaround to avoid error: each word is asociated to regular ranges
                chunck = len(tokenized_sentence_tokens) // len(processed_words)
                for idx, word in enumerate(processed_words):
                    res[word] = range(idx * chunck, min((idx + 1) * chunck, len(tokenized_sentence_tokens)))
                return res

        pointer_token += 1
        end = pointer_token

        res[word] += list(range(start, end))

    # here we arrive at the end of the sentence
    assert tokenized_sentence_tokens[pointer_token] == "</s>"
    end_of_string_position = pointer_token

    # starting word n_gram
    pointer_token += 1
    while pointer_token < len(tokenized_sentence_tokens):
        token = tokenized_sentence_tokens[pointer_token]
        for index_word, word in enumerate(processed_sentence.split()):
            # now, the condition of matching changes: we need to find the word in the token
            if word in token:
                res[word].append(pointer_token)
        pointer_token += 1

    assert pointer_token == len(tokenized_sentence_tokens)
    assert set(sum([v for v in res.values()], [end_of_string_position])) == set(
        range(len(tokenized_sentence_tokens))
    ), print(
        set(range(len(tokenized_sentence_tokens)))
        - set(sum([v for v in res.values()], [end_of_string_position]))
    )  # verify if all tokens are used

    return res


# at text level
def compute_preprocessed_word_score(
    preprocessed_text,
    tokenized_text_tokens,
    scores,
    id_to_token_dicts,
    token_to_id_dicts,
    min_n,
    padding_index=2009603,
    end_of_string_index=0,
):
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

    word_to_score_dicts = []
    word_to_token_idx_dicts = []

    for idx, sentence in enumerate(preprocessed_text):
        tokenized_sentence_tokens = tokenized_text_tokens[idx]  # sentence level, List[str]
        word_to_token_idx = match_word_to_token_indexes(sentence, tokenized_sentence_tokens, min_n)
        score_sentence_topk = scores[idx]  # torch.Tensor, token scores, (top_k, seq_len)

        # Calculate the score for each token and map to words
        word_to_score_topk = []
        for k in range(len(score_sentence_topk)):
            # Initialize word-to-score dictionary with zero values
            word_to_score = {word: 0 for word in sentence.split()}

            score_sentence = score_sentence_topk[k]
            for word, associated_token_idx in word_to_token_idx.items():
                associated_token_idx = torch.tensor(associated_token_idx).int()
                word_to_score[word] = torch.sum(score_sentence[associated_token_idx]).item()

            word_to_score_topk.append(word_to_score.copy())

        word_to_score_dicts.append(word_to_score_topk)
        word_to_token_idx_dicts.append(word_to_token_idx)

    return word_to_score_dicts, word_to_token_idx_dicts


def compute_word_score(word_to_score_dicts, text, n=5, cutoff=0.75):
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

    all_scores_text = []
    mappings = []
    for idx, word_to_score_topk in enumerate(word_to_score_dicts):  # iteration over sentences
        all_scores_topk = []
        processed_words = list(word_to_score_topk[0].keys())
        original_words = text[idx].split()
        original_words = list(filter(lambda x: x != ",", original_words))
        mapping = map_processed_to_original(
            processed_words, original_words, n=n, cutoff=cutoff
        )  # Dict[str, Tuple[List[str], List[float]]]
        mappings.append(mapping)
        for word_to_score in word_to_score_topk:  # iteration over top_k (the preds)
            scores = []
            stopwords_idx = []
            for pos_word, word in enumerate(original_words):
                if word not in mapping:
                    scores.append(0)
                    stopwords_idx.append(pos_word)
                    continue
                matching_processed_word = mapping[word]
                word_score = word_to_score[matching_processed_word]
                scores.append(word_score)

            scores = torch.tensor(scores)
            scores = F.softmax(
                scores, dim=-1
            )  # softmax normalization. Length = len(original_words)
            scores[stopwords_idx] = 0

            all_scores_topk.append(scores)  # length top_k

        all_scores_text.append(all_scores_topk)  # length = len(text)

    return all_scores_text, mappings


def explain_continuous(
    text, processed_text, tokenized_text_tokens, mappings, word_to_token_idx_dicts, all_attr, top_k
):
    """
    Score explanation at letter level.

    Args:
        text (List[str]): List of original sentences.
        processed_text (List[str]): List of preprocessed sentences.
        tokenized_text_tokens (List[List[str]]): List of tokenized sentences.
        mappings (List[Dict[str, str]]): List of mappings from original word to preprocessed word.
        word_to_token_idx_dicts (List[Dict[str, List[int]]]): List of mappings from preprocessed word to token indexes.
        all_attr (torch.Tensor): Tensor of token scores.
        top_k (int): Number of top tokens to consider.

    Returns:
        List[torch.Tensor]: List of letter scores for each sentence.


    """
    all_scores_text = []
    for idx, processed_sentence in enumerate(processed_text):
        tokenized_sentence_tokens = tokenized_text_tokens[idx]
        mapping = mappings[idx]
        word_to_token_idx = word_to_token_idx_dicts[idx]
        original_words = text[idx].split()
        original_words = list(filter(lambda x: x != ",", original_words))

        original_to_token = {}
        original_to_token_idxs = {}

        for original in original_words:
            # original = original.replace(',', '')
            if original not in mapping:
                continue

            matching_processed_word = mapping[original]
            associated_token_idx = word_to_token_idx[matching_processed_word]
            original_to_token[original] = [
                tokenized_sentence_tokens[token_idx] for token_idx in associated_token_idx
            ]
            original_to_token_idxs[original] = associated_token_idx

        scores_for_k = []
        for k in range(top_k):
            scores_for_words = []
            for xxx, original_word in enumerate(original_words):
                original_word_prepro = clean_text_feature([original_word], remove_stop_words=False)[
                    0
                ]

                letters = list(original_word)
                scores_letter = torch.zeros(len(letters), dtype=torch.float32)

                if original_word not in original_to_token:  # if stopword, 0
                    scores_for_words.append(scores_letter)
                    continue

                for pos, token in enumerate(original_to_token[original_word]):
                    pos_token = original_to_token_idxs[original_word][pos]
                    #tok = preprocess_token(token)[0]
                    tok = preprocess_token(token)
                    score_token = all_attr[idx, k, pos_token].item()

                    # Embed the token at the right indexes of the word
                    sm = SequenceMatcher(None, original_word_prepro, tok)
                    a, _, size = sm.find_longest_match()
                    scores_letter[a : a + size] += score_token

                scores_for_words.append(scores_letter)

            all_scores_letter = torch.cat(scores_for_words)
            scores = F.softmax(all_scores_letter, dim=-1)
            scores[all_scores_letter == 0] = 0
            scores_for_k.append(scores)

        scores_for_sentence = torch.stack(scores_for_k)
        all_scores_text.append(scores_for_sentence)

    return torch.stack(all_scores_text)
