import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from explainability.utils import map_processed_to_original


def visualize_word_scores(all_scores, original_texts, pred, n=5, cutoff=0.75):
    for idx, word_to_score_topk in enumerate(all_scores):  # iterate over sentences
        all_scores_topk = all_scores[idx]
        topk = len(all_scores_topk)
        colors = sns.color_palette("mako", n_colors=topk)

        original_words = original_texts[idx].split()

        original_words = list(filter(lambda x: x != ',', original_words))

        for i, word in enumerate(original_words):
            original_words[i] = word.replace(',', '')

        plt.figure(figsize=(16, 6))
        plt.title(f"Word Scores Visualization for Sentence {idx + 1}", fontsize=20, fontweight='bold')

        bar_width = 0.15  # Width of each bar
        indices = np.arange(len(original_words))

        for k in range(topk):
            scores = all_scores_topk[k]
            plt.bar(indices + k * bar_width, scores, bar_width, color=colors[k % len(colors)], label=f'{pred[idx][k]}')

        # Add labels and legend
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Scores', fontsize=12)
        plt.xticks(ticks=indices + bar_width * (topk - 1) / 2, labels=original_words, rotation=45, ha="right", fontsize=10)
        plt.ylim(0, 1)  # Since scores are between 0 and 1 (softmax output)
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()