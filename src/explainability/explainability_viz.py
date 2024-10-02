import matplotlib.pyplot as plt
import difflib
import numpy as np

def map_processed_to_original(processed_words, original_words, n=1, cutoff=0.7):
    # For each word in the original list, find the n closest matching processed words
    word_mapping = {}

    for original_word in original_words:
        processed_word_scores = []

        # Calculate the similarity score for each processed word with the current original word
        for processed_word in processed_words:
            similarity_score = difflib.SequenceMatcher(None, processed_word, original_word).ratio()

            # Only consider matches with similarity above the cutoff
            if similarity_score >= cutoff:
                processed_word_scores.append((processed_word, similarity_score))
        
        # Sort processed words by similarity score in descending order
        processed_word_scores.sort(key=lambda x: x[1], reverse=True)

        # Extract the n closest processed words and their similarity scores
        closest_words = [item[0] for item in processed_word_scores[:n]]
        similarity_scores = [item[1] for item in processed_word_scores[:n]]

        # Add the tuple (list of closest words, list of similarity scores) to the mapping
        word_mapping[original_word] = (closest_words, similarity_scores)

    return word_mapping

def visualize_word_scores(word_to_score_dicts, original_texts, n = 5, cutoff = 0.75):
    for idx, word_to_score in enumerate(word_to_score_dicts):
        processed_words = list(word_to_score.keys())
        original_words = original_texts[idx].split()

        for i, word in enumerate(original_words):
            original_words[i] = word.replace(',', '')
            
        mapping = map_processed_to_original(processed_words, original_words, n=n, cutoff=cutoff)

        scores = []
        for word in original_words:
            processed_words, distances = mapping[word]
            max_score = 0
            for i, potential_processed_word in enumerate(processed_words):
                score = word_to_score[potential_processed_word]
                if score > max_score:
                    max_score = score * distances[i] / np.sum(distances)

            scores.append(max_score)

        # Create the figure and axis
        plt.figure(figsize=(16, 4))
        plt.bar(range(len(original_words)), scores, color='skyblue')

        # Add titles and labels
        plt.title(f"Word Scores for Sentence {idx + 1}", fontsize=16)
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Scores', fontsize=12)

        # Rotate x-axis labels for readability
        plt.xticks(ticks=range(len(original_words)), labels=original_words, rotation=45, ha="right", fontsize=10)
        plt.ylim(0, 1)  # Since scores are between 0 and 1 (softmax output)

        # Show the plot
        plt.tight_layout()  # Adjust layout so everything fits without overlap
        plt.show()
