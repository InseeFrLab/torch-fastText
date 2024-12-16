import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_word_scores(all_scores, original_texts, pred):
    for idx, word_to_score_topk in enumerate(all_scores):  # iterate over sentences
        all_scores_topk = all_scores[idx]
        topk = len(all_scores_topk)
        colors = sns.color_palette("mako", n_colors=topk)

        original_words = original_texts[idx].split()
        original_words = list(filter(lambda x: x != ",", original_words))

        for i, word in enumerate(original_words):
            original_words[i] = word.replace(",", "")

        plt.figure(figsize=(16, 6))
        plt.title(
            f"Word Scores Visualization for Sentence {idx + 1}", fontsize=20, fontweight="bold"
        )

        bar_width = 0.15  # Width of each bar
        indices = np.arange(len(original_words))

        for k in range(topk):
            scores = all_scores_topk[k]
            plt.bar(
                indices + k * bar_width,
                scores,
                bar_width,
                color=colors[k % len(colors)],
                label=f"{pred[idx][k]}",
            )

        # Add labels and legend
        plt.xlabel("Words", fontsize=12)
        plt.ylabel("Scores", fontsize=12)
        plt.xticks(
            ticks=indices + bar_width * (topk - 1) / 2,
            labels=original_words,
            rotation=45,
            fontsize=10,
        )
        plt.ylim(0, 1)  # Since scores are between 0 and 1 (softmax output)
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()


def visualize_letter_scores(all_scores_letters, original_texts, pred):
    topk = len(all_scores_letters)
    for text in original_texts:
        text = [text]
        all_letters = [list(word) for word in text][0]

        for idx, lett in enumerate(all_letters):
            if lett == " ":
                all_scores_letters = np.insert(all_scores_letters, idx, 0, axis=2)

        colors = sns.color_palette("mako", n_colors=topk)
        scores = all_scores_letters
        plt.figure(figsize=(len(all_letters) / 7, 5))

        for k in range(scores.shape[0]):
            res = scores[0, k].cpu().numpy()
            plt.bar(
                range(len(res)),
                res,
                label=pred[0][k],
                width=0.5,
                color=colors[k % len(colors)],
            )
        plt.xticks(range(len(all_letters)), all_letters, rotation=0, fontsize=10)
        plt.legend()
        plt.show()
