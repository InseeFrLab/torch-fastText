import pytest
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torchFastText import torchFastText
from torchFastText.preprocess import clean_text_feature
from torchFastText.datasets import NGramTokenizer

source_path = Path(__file__).resolve()
source_dir = source_path.parent


@pytest.fixture(scope="session", autouse=True)
def data():
    data = {
        "Catégorie": [
            "Politique",
            "Politique",
            "Politique",
            "Politique",
            "Politique",
            "Politique",
            "Politique",
            "Politique",
            "International",
            "International",
            "International",
            "International",
            "International",
            "International",
            "International",
            "International",
            "Célébrités",
            "Célébrités",
            "Célébrités",
            "Célébrités",
            "Célébrités",
            "Célébrités",
            "Célébrités",
            "Célébrités",
            "Sport",
            "Sport",
            "Sport",
            "Sport",
            "Sport",
            "Sport",
            "Sport",
            "Sport",
        ],
        "Titre": [
            "Nouveau budget présenté par le gouvernement",
            "Élections législatives : les principaux candidats en lice",
            "Réforme de la santé : les réactions des syndicats",
            "Nouvelle loi sur l'éducation : les points clés",
            "Les impacts des élections municipales sur la politique nationale",
            "Réforme des retraites : les enjeux et débats",
            "Nouveau plan de relance économique annoncé",
            "La gestion de la crise climatique par le gouvernement",
            "Accord climatique mondial : les engagements renouvelés",
            "Conflit au Moyen-Orient : nouvelles tensions",
            "Économie mondiale : les prévisions pour 2025",
            "Sommet international sur la paix : les résultats",
            "Répercussions des nouvelles sanctions économiques",
            "Les négociations commerciales entre les grandes puissances",
            "Les défis de la diplomatie moderne",
            "Les conséquences du Brexit sur l'Europe",
            "La dernière interview de [Nom de la célébrité]",
            "Les révélations de [Nom de la célébrité] sur sa vie privée",
            "Le retour sur scène de [Nom de la célébrité]",
            "La nouvelle romance de [Nom de la célébrité]",
            "Les scandales récents dans l'industrie du divertissement",
            "Les projets humanitaires de [Nom de la célébrité]",
            "La carrière impressionnante de [Nom de la célébrité]",
            "Les derniers succès cinématographiques de [Nom de la célébrité]",
            "Le championnat du monde de football : les favoris",
            "Record battu par [Nom de l'athlète] lors des Jeux Olympiques",
            "La finale de la Coupe de France : qui remportera le trophée?",
            "Les transferts les plus chers de la saison",
            "Les performances des athlètes français aux championnats du monde",
            "Les nouveaux talents à surveiller dans le monde du sport",
            "L'impact de la technologie sur les sports traditionnels",
            "Les grandes compétitions sportives de l'année à venir",
        ],
    }
    df = pd.DataFrame(data)
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df["Catégorie"])
    df["Titre_cleaned"] = clean_text_feature(df["Titre"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["Titre_cleaned"], y, test_size=0.1, stratify=y
    )
    return X_train, X_test, y_train, y_test


def test_building(data):
    num_tokens = 4
    embedding_dim = 10
    min_count = 1
    min_n = 2
    max_n = 5
    len_word_ngrams = 2
    sparse = False
    vocab_possible_values = [None, [5, 6, 7, 8]]
    cat_embedding_dim_possible_values = [[10, 20, 3, 7], 10, None]
    num_cat_possible_values = [None, 4]
    for vocab, cat_embedding_dim, num_cat in product(
        vocab_possible_values, cat_embedding_dim_possible_values, num_cat_possible_values
    ):
        model = torchFastText(
            num_tokens=num_tokens,
            num_rows=num_tokens,
            embedding_dim=embedding_dim,
            min_count=min_count,
            min_n=min_n,
            max_n=max_n,
            len_word_ngrams=len_word_ngrams,
            sparse=sparse,
            categorical_embedding_dims=cat_embedding_dim,
            categorical_vocabulary_sizes=vocab,
            num_categorical_features=num_cat,
            num_classes=3,
        )
        model._build_pytorch_model()
    assert True, "Model building completed without errors"


def test_training(data):
    num_tokens = 4
    embedding_dim = 10
    min_count = 1
    min_n = 2
    max_n = 5
    len_word_ngrams = 2
    sparse = False
    vocab_possible_values = [None, [5, 6, 7, 8]]
    cat_embedding_dim_possible_values = [[10, 20, 3, 7], 10, None]
    num_cat_possible_values = [None, 4]

    X_train, X_test, y_train, y_test = data

    cat_data_train = np.random.randint(0, 4, (len(X_train), 4))
    cat_data_test = np.random.randint(0, 4, (len(X_test), 4))
    X_train_full = np.concatenate([np.asarray(X_train).reshape(-1, 1), cat_data_train], axis=1)
    X_test_full = np.concatenate([np.asarray(X_test).reshape(-1, 1), cat_data_test], axis=1)
    no_cat_var_model = None
    for vocab, cat_embedding_dim, num_cat in product(
        vocab_possible_values, cat_embedding_dim_possible_values, num_cat_possible_values
    ):
        print(vocab, cat_embedding_dim, num_cat)
        model = torchFastText(
            num_tokens=num_tokens,
            embedding_dim=embedding_dim,
            min_count=min_count,
            min_n=min_n,
            max_n=max_n,
            len_word_ngrams=len_word_ngrams,
            sparse=sparse,
            categorical_embedding_dims=cat_embedding_dim,
            categorical_vocabulary_sizes=vocab,
            num_categorical_features=num_cat,
            num_classes=3,
        )
        if (vocab is None) and (cat_embedding_dim is None) and (num_cat is None):
            model.train(
                np.asarray(X_train),
                np.asarray(y_train),
                np.asarray(X_test),
                np.asarray(y_test),
                num_epochs=1,
                batch_size=32,
                lr=0.001,
                num_workers=4,
            )
            no_cat_var_model = model
        else:
            model.train(
                np.asarray(X_train_full),
                np.asarray(y_train),
                np.asarray(X_test_full),
                np.asarray(y_test),
                num_epochs=1,
                batch_size=32,
                lr=0.001,
                num_workers=4,
            )
        print("done")
    assert True, "Training completed without errors"
    tokenizer = model.tokenizer
    tokenized_text_tokens, tokenized_text, id_to_token_dicts, token_to_id_dicts = (
        tokenizer.tokenize(["Nouveau budget présenté par le gouvernement"])
    )
    assert isinstance(tokenized_text, list)
    assert len(tokenized_text) > 0

    predictions, confidence, all_scores, all_scores_letters = model.predict_and_explain(
        np.asarray(["Nouveau budget présenté par le gouvernement"] + [0] * 4).reshape(1, -1), 2
    )
    assert predictions.shape == (1, 2)

    predictions, confidence, all_scores, all_scores_letters = no_cat_var_model.predict_and_explain(
        np.asarray(["Nouveau budget présenté par le gouvernement"]), 2
    )
    assert predictions.shape == (1, 2)
    # "predictions" contains the predicted class for each input text, in int format. Need to decode back to have the string format
