"""
build_scan_ground_truth.py  (Week 7)
-------------------------------------
Writes data/group1_real/scan_ground_truth.csv: for every (student, short-answer question) on the five
scanned scripts Group 1 digitised, the TEACHER'S red-pen mark and a manual transcription of the
handwriting, both read off the scan PDFs in data/group1_real/scans/.

Why this exists: Group 1's OCR output gives us text but no grades, and our own labels are synthetic.
The scans carry the course teacher's marks, so they are the first REAL human grades the project has.

How the marks were read (and how sure we are):
  * Every short answer is marked out of 2 (no mark on any script exceeds 2; 20 MCQ x 1 + 15 x 2 = 50).
  * The teacher writes "\" for 1, "2" for 2, an alpha-like loop or a cross for 0.
  * CHECK: for every student, MCQ marks + short-answer marks reproduce the circled total on page 1
    exactly (Student_11 40 = 10 + 30, Student_23 23 = 10 + 13, Student_34 33 = 13 + 20,
    Student_35 25 = 11 + 14, Student_50 35 = 13 + 22).
  * Four marks on Student_11's script (Q27, Q31, Q33, Q35) and one on Student_35 (Q33) used a glyph that
    is not a clean numeral; they were resolved to the only value that makes the circled total add up and
    are flagged reading="inferred_from_total".  Re-check them on the scan before quoting single items.
  * Student_11's Q34 runs over two pages and carries a "2" on EACH page (4 in total, which the circled
    total needs). No other script gives Q34 more than 2, so for evaluation the mark is capped at the
    2-mark scale (teacher_mark = 2, teacher_mark_raw = 4).
  * Transcriptions keep the student's spelling and grammar; crossed-out words are dropped.
    They were made by reading the scans and should be spot-checked by a team member.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "group1_real" / "scan_ground_truth.csv"

# circled totals and MCQ sub-totals written by the teacher on page 1
TEACHER_TOTALS = {"Student_11": (40, 10), "Student_23": (23, 10), "Student_34": (33, 13),
                  "Student_35": (25, 11), "Student_50": (35, 13)}

# (question, mark or None if not attempted, reading, transcription)
G = {
"Student_11": [
 ("Q21", 2, "clear", "Supervised Learning is a type of machine learning that are used to train and predict data that are labeled."),
 ("Q22", 1, "clear", "Model M1 is not a good model, despite being able to get a 99% accuracy for train dataset it still fails at the testing dataset. This can occur to certain reasons: - Lack of Dataset - Bad feature Selection - Too many outliers - No Feature Engineering."),
 ("Q23", 1, "clear", "Regression Analysis is the monitoring of the relationship between variables with continuous value."),
 ("Q24", 2, "clear", "2 challenges of Machine Learning: - Data: This is a major problem, even with the best ML Algorithms with bad data the output would also be bad. - Uncertainties: Despite having enough data to train the model, this still could not determine the performance. Since uncertainties may occur, it may effect it's performance."),
 ("Q25", 2, "clear", "2 major applications of data science: - Heart-Failure Prediction - Classifying of Tumor is Dormant or Malignant."),
 ("Q26", 2, "clear", "Structured Data: - Are in tabular format - Has all features that builds up conclusion - format of Excel, CSV. Semi-Structured Data: - Not in tabular format - Has features all mixed - format of image, video, sound."),
 ("Q27", 2, "inferred_from_total", "To find the outlier in the Data we could perform a standard deviation by using the mean as a middle point to determine which data is furthest, the furthest are the outliers."),
 ("Q28", 2, "clear", "Data pre-processing is a stage in the data science life-cycle. During this stage is where we prepare the dataset in order to feed it to the model for training and Testing. In this stage, we perform: - handling miss value - Outlier Detection - Feature Selection - Feature Engineering."),
 ("Q29", 2, "clear", "Data normalization is required in K-NN, because we need to divide the dataset into class or group that has similar property. This will allow the K-NN model to precisely train with labeled data and make better prediction."),
 ("Q30", 2, "clear", "Continuous Attributes are discretized by calculating the mean of the Attribute and measure the distance from one another."),
 ("Q31", 2, "inferred_from_total", "K in K-means algorithms refers to the number of clusters that we will use to divide the dataset into when fitting the model."),
 ("Q32", 2, "clear", "A few hyperparameters of ML Algorithms: - n-estimators - mean - media - neighbors - K - number of clusters - Train-Test set ratio."),
 ("Q33", 2, "inferred_from_total", "Gradient Descent is the derivative value of the dependent variable, where it alters the final output of every calculation with SVM."),
 ("Q34", 4, "two_marks_one_per_page", "Comment on the performance of the Classifiers. Classifier 1: this is a good classifier, whereas there are no mis-classification, able to separate the data properly and has a decent gap between the linear boundaries. Classifier 2: this is not a good classifier. There seems to be 2 different types of data within the linear boundaries gap. Despite having a decent gap length, it stills mis-classify data properly. Classifier 3: this is an ok classifier. Even though there are no mis-classification with the data, this classifier has a rather small gap between the linear boundaries."),
 ("Q35", 2, "inferred_from_total", "Radial Basis Function Kernel work in SVM by measuring and comparing the Distance of the data from the linear boundaries."),
],
"Student_23": [
 ("Q21", 1, "clear", "Supervised learning is a process providing the Input data as well as the correct output data to the Training data set in machine Learning model."),
 ("Q22", 0, "clear", "A model M1 give 99% of accuracy for trainning dataset and give 30% of accuracy for testing dataset. when the Training data set the machine applie the machine learning algorithms and it make the accuracy of Training data set is 99%. when doing the data testing there is not yet applied the machine learning algorithm that why in testing data have only 30% of accuracy."),
 ("Q23", 0, "clear", "Regression analysis is a machine learning Technique use to predict the desired value in the dataset. {1,0}, {spam, no spam}."),
 ("Q24", 2, "clear", "Two Challengs of Machine learning: - Hard to collecting the dataset and use the data to do the processing and make the data cleansing. Challengs to manage the dataset in machine learning process. - Can't give the 100% correct output and have a problem to analyze the output."),
 ("Q25", 2, "clear", "List out any two major application of data science - Healthcare application: help to analyze the health car system. - Finance application: help in the Finance system."),
 ("Q26", 2, "clear", "Distinguish between the terms: structure data and Semi-structure data + structure data: data with a high degrees of organization and easy to understand + Semi-structure data: data with some degrees of organizational."),
 ("Q27", 0, "clear", "to find Outlier in the data we need to do the processing using outlier Technique in Machine learning algorithm: - Detection: Detect the data in the data set. - Imputation: find the outlier data in data set."),
 ("Q28", 2, "clear", "Data pre-processing is a process we are collecting the data, prepare data in the dataset, prepare the dataset for the data training in Machine Learning algorithms."),
 ("Q29", 1, "clear", "Data normalization is required in K-NN because in Data normalization need K-NN algorithm to do the process in Data normalization and it help Data normorlization to make the data easy to do the process."),
 ("Q30", 1, "clear", "The continuous attributes are discretized is because the value of the data in the dataset are in the Quantitative data and in can compresse in a number and all the data are can compress in the number."),
 ("Q31", 2, "clear", "K in K-means algorithms mean the number of cluster If K=2, it mean have 2 cluster, If K=3 it mean it has 3 cluster, and so on."),
 ("Q32", 0, "clear", "Name few hyperparameters of machine learning algorithm - Nearest Neighbor - Naive Baye - KNN - Linear regression - Support vector machine (SVM) - Nuoral Networks - K-mean-Clustering - Association Rules - Q-Learning. - Temporal Difference (TD)."),
 ("Q33", None, "not_attempted", ""), ("Q34", None, "not_attempted", ""), ("Q35", None, "not_attempted", ""),
],
"Student_34": [
 ("Q21", 2, "clear", "Supervised Learning is used for prediction based on Labelled training data. The algorithm learn a function from the input variables (feature) to the output variables (Target)."),
 ("Q22", None, "not_attempted", ""),
 ("Q23", 2, "clear", "Regression analysis is an analysis that used for numerous problem. It predicts the value of a variable based on the value of the other variables."),
 ("Q24", 1, "clear", "two challenges of machine learning: - Data preparation: real life data is messy, need a lot of cleaning. - Data security and privacy"),
 ("Q25", 2, "clear", "List two major application of data science: - Healthcare - bank"),
 ("Q26", 2, "clear", "Distinguish between the term structured and semi-structured data: - Structured data: is the data with high organization. Example: CSV file, spread sheet file - Semi-structured data: is the data with some organization. Example: HTML file, XML file."),
 ("Q27", None, "not_attempted", ""),
 ("Q28", 2, "clear", "Data pre-processing is a process that take raw data and tranform it into a format that can be analyze and understood by a computer."),
 ("Q29", 2, "clear", "Data normalization is required in KNN because data normalization is a process of scaling the value of variable is the value between 0 and 1. data normalization allow the comparetion between the data that has difference unit or scale in KNN."),
 ("Q30", 1, "clear", "The continous attributes are disscretized by convert them using one-hot encoding or use dummy technique."),
 ("Q31", 2, "clear", "K-mean algorithm is an iterative algorithm that tries to partition the dataset into K pre-defined distinct non-overlapping (cluster) where each data point belong to only one group."),
 ("Q32", 2, "clear", "Name few hyperparameters: - Linear kernel - polynomial kernel - Radial kernel."),
 ("Q33", None, "not_attempted", ""),
 ("Q34", 2, "clear", "Comment the performance of the classifiers: Represent them as A, B and C in order: - Classifiers A is the best case: the hyperplanes seperate the data into two classes correctly. The distance between the hyperplanes also great because they are far enough from each other. - Classifiers B is the worst case: the hyperplanes does not seperate the data/object well since there are some remaining data in between the hyperplanes. - Classifier C is good but not the best: the hyperplanes seperate the data/object well but the distance between hyperplanes is too short."),
 ("Q35", None, "not_attempted", ""),
],
"Student_35": [
 ("Q21", 1, "clear", "Supervised Learning is a technique that we used data that trained for training dataset and for testing dataset."),
 ("Q22", 1, "clear", "Comment on model M1: model M1 gives 99% of accuracy for training dataset it means the number of accuracy of M1 model has high percentage of successful. Model M1 gives 30% of accuracy for testing dataset it means M1 model for testing is not enough to become a good model and need more time of training."),
 ("Q23", 0, "clear", "Regression analysis is a process of analysing set of data to enhance training dataset and testing data set which used Linear Regression."),
 ("Q24", 2, "clear", "Elucidate any two challenges of Machine Learning. - model training: as it using dataset for training so it needed us to train model to get more accuracy. - Time Consumption: to make a model to become more accuracy and get successful it might take very long time and more effort."),
 ("Q25", 2, "clear", "List out any two major applications of data science: 1. Stock price prediction 2. Number of Orders prediction"),
 ("Q26", 2, "clear", "Distinguish between terms: structured and semi-structured data: Structured - data well organized - Json, Excel, Spreadsheet - convinience for developer to use for building application - in a specific format that allow us to use for any purpose - easy to manage for training purposes. Semi-structured - data is not well organized - Audio, music - for developer it needs time and effort if consider to use semi-structured data. - It may take time if want to use it as training dataset."),
 ("Q27", 1, "clear", "We find out outlier in the data by using virtualization technique and calculate the distance."),
 ("Q28", 2, "clear", "Data pre-processing is the way of converting raw data into useful data for utilizing purposes."),
 ("Q29", 1, "clear", "Data normalization is required in K-NN because it has big data and need to minimize."),
 ("Q30", None, "not_attempted", ""),
 ("Q31", 1, "clear", "K in K-means algorithm is class."),
 ("Q32", 1, "clear", "Name few hyperparameters of machine learning algorithms: 1. Learning rate alpha 2. Momentum alpha"),
 ("Q33", 0, "inferred_from_total", "Gradient descent is an unsupervised learning for prediction data."),
 ("Q34", None, "not_attempted", ""), ("Q35", None, "not_attempted", ""),
],
"Student_50": [
 ("Q21", 2, "clear", "Supervised learning is a type of learning of Machine learning with the model learn with the label data. and the goal is to find the prediction or an output."),
 ("Q22", 2, "clear", "I think the model M1 will face the issue or error or fault the evaluation. The Model train very accurate but when we test we will see that the Model is not accurate because of the data set we use to test."),
 ("Q23", 1, "clear", "Regression analysis is the analysis that we use on Regression problem to find the relationship between value and the Variable."),
 ("Q24", 2, "clear", "Two challenges of Machine learning: - The accurate of Data set - The model we choose."),
 ("Q25", 2, "clear", "Major application of ML. - Self driving car - prediction application like weather."),
 ("Q26", 2, "clear", "Distinguish between structure and semi-structured data. + structure data. - the data in a proper format row, column or table. Ex. SQL data + Semi-structure data - the data with a structure and format but not in a row or column. Ex. XML,"),
 ("Q27", 2, "clear", "We find out the Outlier in data by using the chart or algorithm to show the relationship between data. Example. Scatter plot. If it is different from other it is a outlier."),
 ("Q28", 2, "clear", "Data pre-processing is a state that we use to clean the data and tranform the data to the accurate and usable data. we do handling the missing data or normalization."),
 ("Q29", 2, "clear", "We need data normalization in K-NN since K-NN we use to point data into group with the closet data. If the data set is big the accurate of grouping or classification is low. So we use use data normalization, It will help the model more accurate."),
 ("Q30", 0, "clear", "Continuous attributes are discretized because the attributes are the data that we can sub devided it."),
 ("Q31", 2, "clear", "K in k-means algorithm is the number of cluster which we devided."),
 ("Q32", 0, "clear", "hyperparameters of machine learning."),
 ("Q33", 0, "clear", "what is gradient descent? - gradient descent is a outlier point which is the data that is out of group."),
 ("Q34", 2, "clear", "Based on the Given linear boundaries, the classification is more accurate when the distance of 2 line is a bit far. In SVM we can extense the distance of 2 parallel line to see if we really classify it accurately or not. So the performance of first and third boundaries is accurate."),
 ("Q35", 1, "clear", "Radial Basis function kernel work in SVM."),
],
}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "question", "attempted", "teacher_mark", "teacher_mark_raw", "max_marks", "mark_reading", "manual_transcription"])
        for sid, rows in G.items():
            short = 0.0
            for q, m, how, text in rows:
                w.writerow([sid, q, "yes" if m is not None else "no", "" if m is None else min(m, 2), "" if m is None else m, 2, how, text])
                short += m or 0
            tot, mcq = TEACHER_TOTALS[sid]
            assert short + mcq == tot, (sid, short, mcq, tot)
    print("wrote", OUT, "- all five circled totals reproduced")


if __name__ == "__main__":
    main()
