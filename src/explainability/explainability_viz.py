import matplotlib.pyplot as plt
import numpy as np
from explainability.utils import map_processed_to_original


def visualize_word_scores(all_scores, original_texts, n=5, cutoff=0.75):
    for idx, word_to_score in enumerate(all_scores):
        scores = list(all_scores[idx].values())
        original_words = original_texts[idx].split()
        for i, word in enumerate(original_words):
            original_words[i] = word.replace(',', '')
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