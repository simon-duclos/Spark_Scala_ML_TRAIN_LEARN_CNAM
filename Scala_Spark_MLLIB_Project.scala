//VERSION AVEC FEATURE ENGINEERING MLLIB
//// lancer avec spark-shell --driver-memory 4G 
//******************Import des bibliothèques*****************
import org.apache.spark.ml.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row,functions}
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.feature.{OneHotEncoder, OneHotEncoderModel}
//
//**************Extraction des données d'apprentissage : 2 fichiers************
//Les commentaires positifs avec ajout d'une colonne "label" = sentiment à 1 et "label_mark" = note parsé d'après le nom de fichier
val train_pos_reviews = spark.read.textFile("./aclImdb/train/pos/*.txt").
		withColumn("label",functions.lit(1)). //Returns Dataset[String](value, label)
                withColumn("_tmp", input_file_name()).withColumn("label_mark", regexp_extract($"_tmp", "_(.*)\\.",1)).drop("_tmp")
//Les commentaires négatifs avec ajout d'une colonne "label" = sentiment à 0 et "label_mark" = note parsé d'après le nom de fichier
val train_neg_reviews = spark.read.textFile("./aclImdb/train/neg/*.txt").
		//map(sentence => sentence.mkString.toLowerCase.split("\\W+")).
		withColumn("label",functions.lit(0)). //Returns Dataset[String](value, label) 
                withColumn("_tmp", input_file_name()).withColumn("label_mark", regexp_extract($"_tmp", "_(.*)\\.",1)).drop("_tmp")
//concaténation des commentaires positifs et négatifs
val train_reviews_temp = train_pos_reviews.union(train_neg_reviews) //Returns Dataset[String] (value, label,label_mark) 
val train_reviews = train_reviews_temp.withColumn("label_mark",col("label_mark").cast(IntegerType))
//**************Extraction des données de test : 2 fichiers************
//Les traitements sont les mêmes que pour les données d'apprentissage (voir ci-haut)
val test_pos_reviews = spark.read.textFile("./aclImdb/test/pos/*.txt").
		withColumn("label",functions.lit(1)). //Returns Dataset[String](value, label) 
                withColumn("_tmp", input_file_name()).withColumn("label_mark", regexp_extract($"_tmp", "_(.*)\\.",1)).drop("_tmp")
val test_neg_reviews = spark.read.textFile("./aclImdb/test/neg/*.txt").
		withColumn("label",functions.lit(0)). //Returns Dataset[String](value, label) 
                withColumn("_tmp", input_file_name()).withColumn("label_mark", regexp_extract($"_tmp", "_(.*)\\.",1)).drop("_tmp")
val test_reviews_temp = test_pos_reviews.union(test_neg_reviews) //Returns Dataset[String] (value, label,label_mark) 
val test_reviews = test_reviews_temp.withColumn("label_mark",col("label_mark").cast(IntegerType))
//**************Pré-traitement (feature engineering)*****************
//****************UTILISATION DES FONCTIONS MLLIB***************
// Tokenizer : découpage des phrases en mots, le tout assemblé dans une liste. Une liste par commentaire
//Le nombre de lignes ne change pas
// Utilisation de regexTokenizer de MLlib pour pouvoir filtrer les interjections - nombreuses - et chiffres divers...
val regexTokenizer = new RegexTokenizer().setPattern("\\W+").setToLowercase(true).setMinTokenLength(3).setInputCol("value").setOutputCol("words")

//Stop words : suppression des mots courts qui sont souvent répétés.
//On utilise la fonction MLlib et le dictionnaire de stopwords par défaut
val swremover = new StopWordsRemover().setInputCol("words").setOutputCol("text")

//Count vectorizer : crée un vecteur contenant chacun des mots utilisés dans tout les verbatims et leurs occurences
//On utilise la fonction de MLlib
val CountVectorizerModel = new CountVectorizer().
		setInputCol("text").
		setOutputCol("features").
		setVocabSize(2000).
		setMinDF(2).
		setBinary(true)
//****************ASSEMBLAGE DES PRE-TRAITEMENTS EN PIPELINE***************
//Pipeline
val fe_pipeline = new Pipeline().setStages(Array(regexTokenizer,swremover,CountVectorizerModel))
val fe_pipeline_model = fe_pipeline.fit(train_reviews)

//Application de la pipeline sur les données d'apprentissage puis de test
val train_reviews_transformed = fe_pipeline_model.transform(train_reviews)
val train = train_reviews_transformed.drop("value").drop("words").drop("text").cache()//.drop("label_mark")
val test = fe_pipeline_model.transform(test_reviews).drop("value").drop("words").drop("text")//.drop("label_mark")
val CountVectorizerModel_size = CountVectorizerModel.getVocabSize//262144
//*******************CHOIX DE L EVALUATION : SENTIMENT (2 CLASSES) OU NOTE (8 CLASSES)****
//val column_to_class = "label_mark"//8 CLASSES
val column_to_class = "label"//2 CLASSES
//*******************ACCURACY EVALUATOR******************
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy").setLabelCol(column_to_class)
//**************CREATION DU MODELE LINEAR SVC (2 CLASSES UNIQUEMENT)**********
if(column_to_class == "label") {
	val linSvc = new LinearSVC().
		setMaxIter(10).
		setFeaturesCol("features").
		setLabelCol(column_to_class).
		setRegParam(0.1)
	val linSvc_model = linSvc.fit(train)//Returns dataframe (label, features, prediction, rawPrediction)
	val train_svc_results = linSvc_model.transform(train)
	val test_svc_results = linSvc_model.transform(test)
	//
	val train_svc_accuracy = evaluator.evaluate(train_svc_results.select("prediction", column_to_class))
	val test_svc_accuracy = evaluator.evaluate(test_svc_results.select("prediction", column_to_class) )
}
//**************CREATION DU MODELE DE REGRESSION LOGISTIQUE***
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
val lr = new LogisticRegression().
	setMaxIter(10).
	setRegParam(0.01).
	setElasticNetParam(0.8).//setFamily("multinomial").
	setLabelCol(column_to_class).
  	setFeaturesCol("features")
val lr_model = lr.fit(train)
val train_lr_results = lr_model.transform(train)
val test_lr_results = lr_model.transform(test)

val train_lr_accuracy = evaluator.evaluate(train_lr_results.select("prediction", column_to_class))
val test_lr_accuracy = evaluator.evaluate(test_lr_results.select("prediction", column_to_class) )

//**************CREATION DU MODELE MLP*****************
if(column_to_class == "label") {
	val layers = Array[Int](CountVectorizerModel_size , 5, 4, 2)
}
else {
	val layers = Array[Int](CountVectorizerModel_size , 5, 11)
}
val mlp_classifier = new MultilayerPerceptronClassifier().
  setLayers(layers).
  setBlockSize(128).
  setSeed(1234L).
  setMaxIter(100).
  setLabelCol(column_to_class).
  setFeaturesCol("features")

// train the model
val mlp_model = mlp_classifier.fit(train)
val train_mlp_results = mlp_model.transform(train)//0.7402
val test_mlp_results = mlp_model.transform(test)//0.73584
//
val train_mlp_accuracy = evaluator.evaluate(train_mlp_results.select("prediction", column_to_class))
val test_mlp_accuracy = evaluator.evaluate(test_mlp_results.select("prediction", column_to_class) )

//**************RESUME DES PERFORMANCES DES MODELES*****************
if(column_to_class == "label"){println(s"SVC : train set accuracy = $train_svc_accuracy ; test set accuracy = $test_svc_accuracy")}
println(s"Logistic regression : train set accuracy = $train_lr_accuracy ; test set accuracy = $test_lr_accuracy")
println(s"MLP : train set accuracy = $train_mlp_accuracy ;  test set accuracy = $test_mlp_accuracy")
