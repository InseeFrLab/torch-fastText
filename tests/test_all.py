import pytest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torchFastText import torchFastText
from torchFastText.preprocess import clean_text_feature

source_path = Path(__file__).resolve()
source_dir = source_path.parent


@pytest.fixture(scope='session', autouse=True)
def data():
    data = {
        'Catégorie': ['Politique', 'Politique', 'Politique', 'Politique', 'Politique', 'Politique', 'Politique', 'Politique',
                    'International', 'International', 'International', 'International', 'International', 'International', 'International', 'International',
                    'Célébrités', 'Célébrités', 'Célébrités', 'Célébrités', 'Célébrités', 'Célébrités', 'Célébrités', 'Célébrités',
                    'Sport', 'Sport', 'Sport', 'Sport', 'Sport', 'Sport', 'Sport', 'Sport'],
        'Titre': [
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
            "Les grandes compétitions sportives de l'année à venir"
        ]
    }
    df = pd.DataFrame(data)
    labelEncoder = LabelEncoder()
    y = labelEncoder.fit_transform(df['Catégorie'])
    df['Titre_cleaned'] = clean_text_feature(df['Titre'])
    X_train, X_test, y_train, y_test = train_test_split(df['Titre_cleaned'], y, test_size=0.1, stratify=y)
    return X_train, X_test, y_train, y_test

@pytest.fixture(scope='session', autouse=True)
def model():
    num_tokens = 4
    embedding_dim = 10
    min_count = 1
    min_n = 2
    max_n = 5
    len_word_ngrams = 2
    sparse = False
    return torchFastText(
        num_tokens=num_tokens,
        embedding_dim=embedding_dim,
        min_count=min_count,
        min_n=min_n,
        max_n=max_n,
        len_word_ngrams=len_word_ngrams,
        sparse=sparse,
    )



def test_model_initialization(model, data):
    assert isinstance(model, torchFastText)
    assert model.num_tokens == 4
    assert model.embedding_dim == 10
    assert model.min_count == 1
    assert model.min_n == 2
    assert model.max_n == 5
    assert model.len_word_ngrams == 2
    assert not model.sparse
    X_train, X_test, y_train, y_test = data
    model.train(
        np.asarray(X_train),
        np.asarray(y_train),
        np.asarray(X_test),
        np.asarray(y_test),
        num_epochs=1,
        batch_size=32,
        lr=0.001,
        num_workers=4
    )
    assert True, "Training completed without errors"
    tokenizer = model.tokenizer
    tokenized_text_tokens, tokenized_text, id_to_token_dicts, token_to_id_dicts= tokenizer.tokenize(["Nouveau budget présenté par le gouvernement"])
    assert isinstance(tokenized_text, list)
    assert len(tokenized_text) > 0
    #assert "gouvern </s>" in tokenized_text_tokens[0]
    predictions, confidence, all_scores, all_scores_letters = model.predict_and_explain(np.asarray(["Nouveau budget présenté par le gouvernement"]), 2)
    assert predictions.shape == (1, 2)
    # "predictions" contains the predicted class for each input text, in int format. Need to decode back to have the string format
    
