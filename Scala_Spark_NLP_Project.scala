//// lancer avec spark-shell --driver-memory 4G --packages JohnSnowLabs:spark-nlp:2.2.2
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
//
//**************Extraction des données d'apprentissage : 2 fichiers************
//Les traitements sont les mêmes que pour les données d'apprentissage (voir ci-haut)
val test_pos_reviews = spark.read.textFile("./aclImdb/test/pos/*.txt").
		withColumn("label",functions.lit(1)). //Returns Dataset[String](value, label) 
                withColumn("_tmp", input_file_name()).withColumn("label_mark", regexp_extract($"_tmp", "_(.*)\\.",1)).drop("_tmp")
val test_neg_reviews = spark.read.textFile("./aclImdb/test/neg/*.txt").
		withColumn("label",functions.lit(0)). //Returns Dataset[String](value, label) 
                withColumn("_tmp", input_file_name()).withColumn("label_mark", regexp_extract($"_tmp", "_(.*)\\.",1)).drop("_tmp")
val test_reviews_temp = test_pos_reviews.union(test_neg_reviews) //Returns Dataset[String] (value, label,label_mark) 
val test_reviews = test_reviews_temp.withColumn("label_mark",col("label_mark").cast(IntegerType))
//**************Pré-traitement (feature engineering*****************
//**************FEATURE ENGINEERING with JohnSnowLab NLP*****************
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
import com.johnsnowlabs.nlp.SparkNLP
import com.johnsnowlabs.nlp.Finisher
val explainPipelineModel = PretrainedPipeline("explain_document_ml", lang="en").model
val finisherExplain = new Finisher().setInputCols("token", "lemma", "pos", "stem")
val nlp_pipeline = new Pipeline().setStages(Array(explainPipelineModel,finisherExplain))

val nlp_pipeline_model = nlp_pipeline.fit(train_reviews.withColumnRenamed("value","text"))
val annotation = nlp_pipeline_model.transform(train_reviews.withColumnRenamed("value","text"))
val annotation_test = nlp_pipeline_model.transform(test_reviews.withColumnRenamed("value","text"))
val an = annotation.select("text","label","label_mark","finished_lemma")
val an_test = annotation_test.select("text","label","label_mark","finished_lemma")
val CountVectorizerModel = new CountVectorizer().setInputCol("finished_lemma").setOutputCol("features").setMinDF(2).setBinary(true).setVocabSize(2000)
val tt = CountVectorizerModel.fit(an)
val train = tt.transform(an)
val test = tt.transform(an_test)

val CountVectorizerModel_size = CountVectorizerModel.getVocabSize

val column_to_class = "label_mark"//8 CLASSES
//val column_to_class = "label"//2 CLASSES
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
