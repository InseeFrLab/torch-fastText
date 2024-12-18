import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from torchFastText import torchFastText

source_path = Path(__file__).resolve()
source_dir = source_path.parent


class tftTest(unittest.TestCase):
    def __init__(self, methodName="runTest"):
        super().__init__(methodName)
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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df['Titre'], y, test_size=0.1, stratify=y)
        

    

    def test_train_no_categorical_variables(self):
        num_buckets = 4
        embedding_dim = 10
        min_count = 1
        min_n = 3
        max_n = 6
        len_word_ngrams = 10
        sparse = False
        self.torchfasttext = torchFastText(
            num_buckets=num_buckets,
            embedding_dim=embedding_dim,
            min_count=min_count,
            min_n=min_n,
            max_n=max_n,
            len_word_ngrams=len_word_ngrams,
            sparse=sparse,
        )
        self.torchfasttext.train(
            np.asarray(self.X_train),
            np.asarray(self.y_train),
            np.asarray(self.X_test),
            np.asarray(self.y_test),
            num_epochs=2,
            batch_size=32,
            lr=0.001
        )
        self.assertTrue(True, msg="Training Validated")
        #print(self.torchfasttext.predict(np.asarray(["Star John elected president"]), 3))

        #self.assertTrue(True, msg="Predicted")



if __name__ == "__main__":
    unittest.main()
