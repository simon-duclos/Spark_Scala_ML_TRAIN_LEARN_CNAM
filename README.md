Analyse de Sentiment avec Spark
Simon DUCLOS 
Sujet :
L’objectif est de prédire la note (entre 1 et 10) à partir du texte de la revue d’un film, pour les données issues de la base de données IMDB (50000 revues sur des films, données accessibles sur http://ai.stanford.edu/~amaas/data/sentiment/) ; découper l’ensemble de données en données d’apprentissage et données de test.

Table des matières :

Contents
0.    Introduction	
I.	Etude préalable des données	
  1.	Présentation des données
  2.	Répartition
  3.	Remarque sur l’exploration des données
II.	Etapes de traitement	
  1.	Séparer la base d’apprentissage de la base de test	
  2.	Obtenir les sentiments et notes des verbatims (labels)	
  3.	Prétraitement des données texte (feature engineering)	
    a.	Pré-traitement avec les transformateurs de MLlib	
    b.	Pré-traitement avec pipeline Johnsnow labs puis vectorisation	
  4.	Classifieurs	
III.	Comparatif des résultats	
IV.	Difficultés rencontrées	
IV.	Conclusion

 
0.    Introduction
La base de données « Large Movie Review dataset» provient de l’extraction d’une partie des données du site IMDB, référence aux Etats-Unis (et dans le monde) pour les avis des spectateurs sur les films. IMDB est site commercial qui fait partie du groupe Amazon.
Cette base de données a été présentée pour la première fois en 2011 dans un article de recherche sur l’analyse de sentiments [1]. Elle présente 50 000 verbatims de commentaires de film ainsi que la note donnée au film par les spectateurs.
Cette base est aujourd’hui publique et est hébergée sur le site de l’université de Stanford.

 
I.	Etude préalable des données
Remarque générale : les commentaires des films présents dans la base de données seront désignés par “Verbatims” ou “Commentaires”, c’est à dire un commentaire d’un spectateur concernant un film et auquel est associé une note.
1.	Présentation des données	

La base de données contient 50 000 lignes correspond chacune à un verbatim, soit :
-	25 000 données d’apprentissage
o	12 500 données de sentiment « négatif » note entre 0 et 4
o	12 500 données de sentiment « positif » entre 6 et 9
-	25 000 données de test
o	12 500 données de sentiment « négatif » entre 0 et 4
o	12 500 données de sentiment « positif » entre 6 et 9
A noter que les commentaires associés à une note 5 ou 6 sont exclus de la base de données.
Dans le cadre de cette étude, je présenterai 2 types de résultats :
•	Les résultats par note
•	Les résultats par sentiment “positif” ou “négatif”, ce groupement étant une agrégation du premier.
2.	Répartition

L’objectif de cette étude étant de prédire la note, la première étape de la découverte des données est de regarder la répartition de celles-ci.

Bonne nouvelle ! L'échantillonnage a déjà été réalisé en amont, la base de test et de train contiennent un nombre de verbatims proche pour chaque note. Le coefficient de corrélation (Pearson) est de 0,99. Les bases d’apprentissage et de test sont issues de la même distribution, nous pouvons donc nous servir de la première pour apprendre, la deuxième pour généraliser.
En revanche, si les classes de sentiment sont équilibrées (12500 exemples dans chaque), ça n’est clairement pas le cas des classes de note (surpondération de 1 et 10 qui comptent en moyenne deux fois plus de verbatim que les autres classes)


Après avoir regardé la répartition des classes à prédire, intéressons-nous aux verbatims et aux mots.

3.	Remarque sur l’exploration des données

N’ayant pas trouvé de moyen “simple” de visualiser des données sous Spark avec Scala (contrairement à des langages comme python qui possède des librairies comme seaborn), j’ai extrait des données directement depuis Spark et fait les graphes et les calculs (moyenne, coefficient de régression) directement dans Excel, car c’est le logiciel-proposé par le CNAM- le plus simple à utiliser. 
Suite au cours sur la visualisation de données, et en particulier le classement des variables rétiniennes de Bertin, les 3 graphiques précédents exploitent la taille et la valeur (préférée ici à la couleur).

II.	Etapes de traitement

Détaillons ici les étapes du traitement, de l’extraction au modèle prédictif final.
A noter que j’ai choisi de travailler exclusivement avec la version “ml” de MLlib (org.apache.spark.ml), car elle est postérieure - et recommandée- par Spark et qu’elle permet de ne travailler uniquement qu’avec des datasets, évitant des conversions RDD-dataframe couteuses.

1.	Séparer la base d’apprentissage de la base de test

Le modèle de données proposées contient déjà une dissociation Train/Test que nous réutiliserons : 
L’extraction des données depuis le site crée automatiquement un répertoire « Train » et « Test ». Les fichiers dans les sous-répertoires de Train seront associés à la base d’apprentissage et ceux du sous-répertoire de Test à la base de test. 

2.	Obtenir les sentiments et notes des verbatims (labels)

Pour les sentiments (positifs ou négatifs), les verbatims sont eux aussi rassemblés dans les répertoires distincts  « pos » et « neg ». Nous réutilisons cette séparation.
La différentiation en classe de note s’appuiera en revanche sur un parsing du nom de fichier, chaque verbatim étant présent dans un fichier de nomenclature précise : NUMEROVERBATIM_NOTE
Le parsing s’opère par identification des données régulières (regex), voici la formule utilisée en scala :
withColumn("_tmp", input_file_name()).withColumn("label_mark", regexp_extract($"_tmp", "_(.*)\\.",1)).drop("_tmp")


 
3.	Prétraitement des données texte (feature engineering)

En s’inspirant des TP, deux types de prétraitement ont été testés en parallèle :
•	Un ensemble d’opérations de pré-traitement propres à MLlib
•	Un pipeline pré-existant de John-Snow labs
a.	Pré-traitement avec les transformateurs de MLlib

Les pré-traitements suivants ont été effectués : 
•	Découpage des verbatims en suite de mots en utilisant la fonction MLlib regexTokenizer : grace au regex “W+”, on ne garde que les caractères type lettre, ce qui permet d’éliminer les nombres (qui après observation manuelle d’un certain nombre de verbatims, servent principalement à séparer des paragraphes dans les verbatims longs et ne sont d’aucune utilité décisionnelle), et aussi de se séparer des caractères de ponctuation (mais après observation de quelques verbatims positifs et négatifs, on trouve dans les 2 points d’exclamation et d’interrogation). On ne se concentrera ici que sur les mots.
•	Filtrage des stop words. Comme précédemment, la fonction utilisée est StopWordsRemover de MLlib, avec la liste de stopwords par défaut.
•	Vectorisation en utilisant CountVectorizer : il s’agit d’une vectorisation simple : chaque nouveau mot générant un vecteur.  Cela génère énormément de colonnes (>44000 correspondants à des mots différents, mot pouvant être des occurrences différentes du même mot (au singulier, au pluriel), des conjugaisons du même verbe). On fera l’essai avec 44000 colonnes (représentation vectorielle extrêmement creuse) et avec 2000 colonnes des mots les plus fréquents (le filtage est nativement proposé par CountVectorizer).
•	Vectorisation avec word2vec (à la place de CountVectorizer). Comme vu en cours, l’utilisation de word2vec génère un vecteur court (100 colonnes dans le TP), et est adapté aux petites séquences de mot (10 à 15). Ici vu la taille des verbatims, le résultat était très en dessous de la vectorisation par CountVectorizer.  A noter que j’ai dû recréer un fichier word2vec.ml à partir de ml.feature.Word2Vec pour travailler avec MLlib car celui fourni en TP provenait de mllib.feature.Word2VecModel et n’était pas compatible avec ml.
J’ai assemblé les traitements dans un pipeline pour plus de clarté dans le code.

b.	Pré-traitement avec pipeline Johnsnow labs puis vectorisation

J’ai choisi de travailler avec le pipeline “explain_document_ml” car il a l’avantage de contenir les opérations de lemmatisation et racinisation. Il contient aussi les opérations de filtrage des stopwords. 
En sortie, j’ai choisi de travailler avec les lemmes. Ceux-ci ont été vectorisés avec CountVectorizer, ce qui m’a permis de filtrer les mots de plus de 2 caractères. Je n’ai en effet pas trouvé dans la documentation de la pipeline le moyen de le faire directement.


4.	Classifieurs

Avec les verbatims pré-traités et vectorisés, il est possible d’appliquer différentes familles d’estimateurs pour calculer la note ou le sentiment.
J’ai choisi 3 familles qui existent dans MLlib, et qui donc contiennent déjà une capacité à travailler avec des données massives car parallélisables :
•	SVM linéaires (LinearSVC - les fonctions à noyaux ne sont pas implémentées dans MLlib) comme vu en TP pour le classement binaire uniquement, car la séparation multi-classes n’est pas triviale dans MLlib et que les deux familles suivantes apportent déjà de bons résultats.
•	Régression logistique (LogisticRegression)
•	Perceptron multi-couches(MultilayerPerceptronClassifier) 
J’ai appliqué une validation croisée avec CrossValidator sur SVM et regression logistique (trop long sur les MLP) afin de pouvoir sélectionner les meilleurs hyperparamètres, appliquée sur 10% de l’échantillon d’apprentissage supposé représentatif (pour des raisons pratiques liées au fait que son mon ordinateur personnel, entraîner un modèle SVM ou MLP prend environ 5 minutes par hyperparamètres, ce qui rend une évaluation de 10 paramètres très longue sur la totalité de l’échantillon.
La fonction de perte utilisée est l’entropie croisée, binaire dans le cas du classement du sentiment, multinomiale dans le cas de la prédiction de la note.
Quant à la métrique d’évaluation, j’ai utilisé l’exactitude  (accuracy).

 
III.	Comparatif des résultats

Les résultats que présentés ci-bas sont le résultat des configurations suivantes : 
•	Préprocessing MLlib avec word2vec (de 100 colonnes) > puis régression logistique
•	Préprocessing MLlib avec CountVectorizer (autant de colonnes que de mots) > puis régression logistique
•	Préprocessing MLlib avec CountVectorizer (2000 colonnes correspondant aux 2000 mots les plus utilisés) > puis régression logistique ou SVM
•	Préprocessing Spark NLP (JohnSnowlab) puis CountVectorizer (2000 colonnes)

Voici les résultats obtenus sur 2 classes. Il s’agit de la justesse (accuracy, voir ci-haut) en classification binaire :
 

Sur les 8 classes de notes, même métrique en classification multi-classes :
 
En raison de problèmes de performances, je n’ai pas pu utiliser le classement par perceptron multi-couches dans la plupart des cas (sauf sur countvectorizer 2000 colonnes, mais le traitement a pris plus d’une heure).
Et pour la classification multi-classes, le SVM demande des développements supplémentaires, non effectué ici dans le cadre de ce projet.


IV.	Difficultés rencontrées

Les difficultés rencontrées dans ce projet sont nombreuses et peuvent être classées en 2 catégories :
•	Les problèmes de typage et de conversion de type
•	La faiblesse des messages d’erreur de Spark
J’avais commencé à travailler initialement avec des RDD, que j’ai cherché à convertir en datasets pour appliquer les fonctions MLlib, mais finalement le choix le plus judicieux a été de ne travailler qu’avec des datasets. En outre, la conversion de types dans Spark n’est jamais triviale : par exemple essayer de compter le nombre de mots d’une colonne d’un dataset ne peut pas se faire de la même façon que compter le nombre de mots d’un dataset ne contenant qu’une colonne.
Au final pour les extractions de données, j’ai choisi d’effectuer des requêtes SQL via spark.sql plutôt que d’utiliser les fonctions natives de Spark...
Le deuxième point concerne les messages d’erreur de Spark, plutôt difficiles à déchiffrer. Par exemple la “Java heap space error” lors de la tentative de génération d’un modèle word2vec de plus de 100 colonnes. Aussi la “Failed to execute user defined fonction [..] one hot encoder” qui signifie (après recherche) que le nombre de paramètres en couche de sortie du perceptron ne correspond pas aux nombre de valeurs différentes de la classe d’observation (il faut 11 couches en sortie pour les classes de notes 1, 2, 3, 4, 7, 8,9,10 car il y a toujours 0 par défaut et il faut compter les classes absentes 5,6...).
Enfin, pour certains traitements de données, la mémoire vive ne suffit pas et Spark écrit la mémoire sur le disque sous forme de rdd. Les traitements s’éternisent et je n’ai pas trouver comment estimer la durée, arrêter les traitements, d’avoir un plan de transactions...
Pour finir, mentionnons la non répétabilité des temps de traitement : du simple au quintuple, sans que je ne comprenne vraiment pourquoi.
IV.	Conclusion

En utilisant les fonctions de la bibliothèque Mllib de Spark sous scala, il est possible d’obtenir une classification “correcte” des verbatims lors de la séparation en 2 classes avec 87% d’accuracy. En ce qui concerne la prédiction des notes en elles-mêmes, la prédiction est beaucoup moins précise, avec seulement 43% de justesse. 
J’ai choisi de me concentrer sur la création d’un modèle de bout en bout (end-to-end) en essayant différentes fonctions de classification non linéaires (MLP, SVM, régression logistique). 
Afin d’améliorer la justesse du modèle, un travail pourrait intégrer d’autres familles de classifieurs (les arbres par exemple), tester d’autres modèles vectoriels de texte (pourquoi pas des embeddings type BERT présents dans Spark NLP), et bien sûr de retravailler les métriques d’apprentissage pour prendre en compte la répartition déséquilibrée des classes de note. Et pour la gestion de données massives, un stockage hdfs et une parallélisation des traitements sur plusieurs serveurs permettraient à coup sûr d’améliorer la rapidité des traitements.
 
Références 
[1] author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher}




