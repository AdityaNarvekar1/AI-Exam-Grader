"""
generate_dataset.py
--------------------
Group 2 - AI Exam Answer Grader
Builds a synthetic, rubric-anchored exam-answer grading dataset covering five
core Data Science engineering subjects. Since no real institutional answer
scripts were available (Week-1 decision: synthetic dataset), each record is
hand-authored so that the "ground-truth" human_score is anchored to an
explicit rubric, not an arbitrary number.

Schema (matches the platform-wide Section 10 I/O philosophy):
    qid, subject, question, reference_answer, rubric (json), max_marks,
    student_id, student_answer, points_covered (json), human_score,
    quality_tier, split
"""

import json
import random
import csv
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).resolve().parents[2] / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. QUESTION BANK (5 subjects x 3 questions), each with an examiner-style
#    reference answer and a rubric that assigns marks to discrete concepts.
# ---------------------------------------------------------------------------

QUESTIONS = [
    # ---------------- Python Programming for Data Science ----------------
    {
        "qid": "PY01", "subject": "Python Programming",
        "question": "Explain the key differences between a list and a tuple in Python.",
        "max_marks": 5,
        "reference_answer": (
            "A list is mutable, meaning its elements can be changed, added, or removed "
            "after creation, whereas a tuple is immutable and cannot be modified once "
            "created. Lists are defined with square brackets and tuples with parentheses. "
            "Because tuples are immutable, they are generally faster to iterate over and "
            "can be used as dictionary keys or set elements, while lists cannot."
        ),
        "rubric": [
            {"id": "p1", "text": "A list is mutable, meaning it can be changed after creation, while a tuple is immutable and cannot be changed", "marks": 2,
             "binding": {"subject_terms": ["list"], "correct_value_terms": ["mutable"], "contradicting_value_terms": ["immutable"]}},
            {"id": "p2", "text": "Lists are written using square brackets, while tuples are written using parentheses", "marks": 1},
            {"id": "p3", "text": "Tuples are faster and hashable, so they can be used as dictionary keys, unlike lists", "marks": 2},
        ],
    },
    {
        "qid": "PY02", "subject": "Python Programming",
        "question": "What is a Python generator and why is it memory-efficient?",
        "max_marks": 5,
        "reference_answer": (
            "A generator is a function that uses the 'yield' keyword to produce a "
            "sequence of values lazily, one at a time, instead of returning them all at "
            "once. It maintains its internal state between calls, so the next value is "
            "computed only when requested. This makes generators memory-efficient because "
            "the entire sequence is never stored in memory simultaneously, unlike a list."
        ),
        "rubric": [
            {"id": "p1", "text": "A generator uses the yield keyword to produce values lazily, one at a time", "marks": 2},
            {"id": "p2", "text": "A generator maintains its internal state between successive calls", "marks": 1},
            {"id": "p3", "text": "Generators are memory-efficient because the full sequence of values is never stored in memory at once", "marks": 2},
        ],
    },
    {
        "qid": "PY03", "subject": "Python Programming",
        "question": "Explain how NumPy broadcasting works, with an example.",
        "max_marks": 10,
        "reference_answer": (
            "Broadcasting is the mechanism NumPy uses to perform element-wise operations "
            "on arrays of different shapes by automatically expanding the smaller array "
            "along dimensions of size 1, without actually copying data. Two dimensions "
            "are compatible for broadcasting if they are equal or one of them is 1. For "
            "example, adding a (3,) array to a (3,3) array broadcasts the smaller array "
            "across every row of the larger array, so each row gets the same vector added "
            "to it. This avoids explicit loops and is computationally efficient."
        ),
        "rubric": [
            {"id": "p1", "text": "Broadcasting lets NumPy perform element-wise operations on arrays that have different shapes", "marks": 3},
            {"id": "p2", "text": "Two dimensions are compatible for broadcasting when they are equal or when one of them is 1", "marks": 3},
            {"id": "p3", "text": "A correct example is given that shows how a smaller array is broadcast across a larger array", "marks": 2},
            {"id": "p4", "text": "Broadcasting is efficient because it avoids writing an explicit loop and does not copy the data", "marks": 2},
        ],
    },

    # ---------------------- Probability & Statistics -----------------------
    {
        "qid": "ST01", "subject": "Probability & Statistics",
        "question": "State and explain the Central Limit Theorem (CLT).",
        "max_marks": 10,
        "reference_answer": (
            "The Central Limit Theorem states that the sampling distribution of the "
            "sample mean approaches a normal distribution as the sample size becomes "
            "large, regardless of the shape of the population distribution, provided the "
            "population has a finite variance. In practice, this holds reasonably well "
            "for sample sizes of about 30 or more. The CLT is important because it allows "
            "us to use normal-distribution-based inference (confidence intervals, "
            "hypothesis tests) even when the underlying population is not normal."
        ),
        "rubric": [
            {"id": "p1", "text": "The sampling distribution of the sample mean tends toward a normal distribution", "marks": 3},
            {"id": "p2", "text": "This holds true regardless of the shape of the population distribution, as long as the population has finite variance", "marks": 3},
            {"id": "p3", "text": "A common practical rule of thumb is that this approximation holds well once the sample size is around 30 or more", "marks": 2},
            {"id": "p4", "text": "This theorem matters because it allows normal-distribution-based inference even when the underlying population is not normal", "marks": 2},
        ],
    },
    {
        "qid": "ST02", "subject": "Probability & Statistics",
        "question": "Differentiate between Type I and Type II errors in hypothesis testing.",
        "max_marks": 5,
        "reference_answer": (
            "A Type I error occurs when the null hypothesis is rejected even though it "
            "is actually true (a false positive), and its probability is denoted by "
            "alpha, the significance level. A Type II error occurs when the null "
            "hypothesis is not rejected even though it is actually false (a false "
            "negative), and its probability is denoted by beta."
        ),
        "rubric": [
            {"id": "p1", "text": "A Type I error occurs when a true null hypothesis is incorrectly rejected, which is a false positive with probability alpha", "marks": 2.5,
             "binding": {"subject_terms": ["type i", "type 1"], "correct_value_terms": ["false positive"], "contradicting_value_terms": ["false negative"]}},
            {"id": "p2", "text": "A Type II error occurs when a false null hypothesis is not rejected, which is a false negative with probability beta", "marks": 2.5,
             "binding": {"subject_terms": ["type ii", "type 2"], "correct_value_terms": ["false negative"], "contradicting_value_terms": ["false positive"]}},
        ],
    },
    {
        "qid": "ST03", "subject": "Probability & Statistics",
        "question": "Explain the difference between correlation and causation, with an example.",
        "max_marks": 5,
        "reference_answer": (
            "Correlation measures the strength and direction of a statistical "
            "association between two variables, but it does not imply that one variable "
            "causes the other. Causation means that a change in one variable directly "
            "produces a change in another. For example, ice cream sales and drowning "
            "incidents are correlated because both increase in summer, but ice cream "
            "sales do not cause drownings; a third variable, hot weather, drives both."
        ),
        "rubric": [
            {"id": "p1", "text": "Correlation only measures a statistical association between two variables", "marks": 2},
            {"id": "p2", "text": "Causation means that one variable directly causes a change in another variable", "marks": 1.5},
            {"id": "p3", "text": "A correct example is given that includes a confounding variable explaining the correlation", "marks": 1.5},
        ],
    },

    # ----------------------------- Machine Learning -------------------------
    {
        "qid": "ML01", "subject": "Machine Learning",
        "question": "Explain the bias-variance tradeoff.",
        "max_marks": 10,
        "reference_answer": (
            "Bias is the error introduced by approximating a real-world problem with a "
            "simplified model, and high bias causes underfitting. Variance is the error "
            "introduced by the model's sensitivity to small fluctuations in the training "
            "data, and high variance causes overfitting. The bias-variance tradeoff "
            "refers to the fact that decreasing bias (using a more complex model) "
            "typically increases variance, and vice versa, so the goal is to find a model "
            "complexity that minimizes total error, which is the sum of bias-squared, "
            "variance, and irreducible error."
        ),
        "rubric": [
            {"id": "p1", "text": "Bias is defined as the error from an overly simple model, which leads to underfitting", "marks": 2.5},
            {"id": "p2", "text": "Variance is defined as the error from a model that is too sensitive to the training data, which leads to overfitting", "marks": 2.5},
            {"id": "p3", "text": "There is a tradeoff between bias and variance, where reducing one tends to increase the other", "marks": 3,
             "cues": {"confirming_phrases": ["tradeoff", "trade off", "trade-off", "inversely", "one increases", "one decreases", "reducing one"],
                       "contradicting_phrases": ["always happen together", "increase together", "both increase", "both get worse", "happen together", "both go up", "both go down"]}},
            {"id": "p4", "text": "The goal is to choose a model complexity that minimizes the total error by balancing bias and variance", "marks": 2},
        ],
    },
    {
        "qid": "ML02", "subject": "Machine Learning",
        "question": "What is overfitting, and how can regularization help prevent it?",
        "max_marks": 10,
        "reference_answer": (
            "Overfitting occurs when a model learns the noise and specific patterns of "
            "the training data too closely, resulting in excellent training performance "
            "but poor generalization to unseen data. Regularization addresses this by "
            "adding a penalty term to the loss function that discourages overly complex "
            "models or large coefficient values. L1 (Lasso) regularization can shrink "
            "some coefficients to exactly zero, performing feature selection, while L2 "
            "(Ridge) regularization shrinks coefficients smoothly without necessarily "
            "making them zero. Both reduce variance at the cost of a small increase in "
            "bias."
        ),
        "rubric": [
            {"id": "p1", "text": "Overfitting occurs when a model fits the noise in the training data too closely, resulting in poor generalization to new data", "marks": 3},
            {"id": "p2", "text": "Regularization adds a penalty term to the loss function that discourages large or overly complex coefficients", "marks": 3},
            {"id": "p3", "text": "L1 regularization can shrink some coefficients exactly to zero, while L2 regularization shrinks coefficients smoothly without making them exactly zero", "marks": 2},
            {"id": "p4", "text": "Regularization reduces variance at the cost of a small increase in bias", "marks": 2},
        ],
    },
    {
        "qid": "ML03", "subject": "Machine Learning",
        "question": "Explain the difference between bagging and boosting.",
        "max_marks": 10,
        "reference_answer": (
            "Bagging (Bootstrap Aggregating) trains multiple models in parallel on "
            "different bootstrap samples of the training data and combines their "
            "predictions by averaging or majority voting, which primarily reduces "
            "variance; Random Forest is a classic example. Boosting trains models "
            "sequentially, where each new model focuses on correcting the errors made by "
            "the previous ones, typically by reweighting misclassified samples, which "
            "primarily reduces bias; AdaBoost and Gradient Boosting are examples."
        ),
        "rubric": [
            {"id": "p1", "text": "Bagging trains multiple models in parallel on different bootstrap samples of the data, which reduces variance", "marks": 4},
            {"id": "p2", "text": "Boosting trains models sequentially, with each new model correcting the errors of the previous ones, which reduces bias", "marks": 4},
            {"id": "p3", "text": "A correct example is given for each, such as Random Forest for bagging and AdaBoost or Gradient Boosting for boosting", "marks": 2},
        ],
    },

    # ---------------------- Database Management Systems ---------------------
    {
        "qid": "DB01", "subject": "Database Management Systems",
        "question": "Explain normalization and briefly describe 1NF, 2NF and 3NF.",
        "max_marks": 10,
        "reference_answer": (
            "Normalization is the process of organizing relational database tables to "
            "reduce data redundancy and avoid update, insertion, and deletion anomalies. "
            "First Normal Form (1NF) requires that all attribute values be atomic, with "
            "no repeating groups. Second Normal Form (2NF) requires 1NF and that every "
            "non-key attribute be fully functionally dependent on the whole primary key, "
            "eliminating partial dependency. Third Normal Form (3NF) requires 2NF and "
            "that no non-key attribute depends on another non-key attribute, eliminating "
            "transitive dependency."
        ),
        "rubric": [
            {"id": "p1", "text": "The purpose of normalization is to reduce data redundancy and avoid update, insertion, and deletion anomalies", "marks": 2},
            {"id": "p2", "text": "First Normal Form requires that all attribute values be atomic, with no repeating groups", "marks": 2.5},
            {"id": "p3", "text": "Second Normal Form requires that there be no partial dependency of a non-key attribute on part of the primary key", "marks": 2.5},
            {"id": "p4", "text": "Third Normal Form requires that there be no transitive dependency between non-key attributes", "marks": 3},
        ],
    },
    {
        "qid": "DB02", "subject": "Database Management Systems",
        "question": "What is a JOIN in SQL? Explain INNER JOIN vs LEFT JOIN with an example.",
        "max_marks": 10,
        "reference_answer": (
            "A JOIN combines rows from two or more tables based on a related column "
            "between them. An INNER JOIN returns only the rows where there is a match in "
            "both tables. A LEFT JOIN returns all rows from the left table and the "
            "matched rows from the right table, filling in NULLs for right-table columns "
            "when there is no match. For example, joining a Students table with an "
            "Enrollments table using INNER JOIN would return only students who have at "
            "least one enrollment, whereas a LEFT JOIN would also include students with "
            "zero enrollments."
        ),
        "rubric": [
            {"id": "p1", "text": "A JOIN combines rows from two or more tables based on a related column between them", "marks": 2},
            {"id": "p2", "text": "An INNER JOIN returns only the rows that have matching values in both tables", "marks": 3},
            {"id": "p3", "text": "A LEFT JOIN returns all rows from the left table along with NULL values for unmatched rows from the right table", "marks": 3},
            {"id": "p4", "text": "A correct illustrative example is given showing the difference between an INNER JOIN and a LEFT JOIN", "marks": 2},
        ],
    },
    {
        "qid": "DB03", "subject": "Database Management Systems",
        "question": "Explain the ACID properties of a database transaction.",
        "max_marks": 10,
        "reference_answer": (
            "ACID stands for Atomicity, Consistency, Isolation, and Durability. "
            "Atomicity ensures a transaction is all-or-nothing: either every operation "
            "completes or none do. Consistency ensures a transaction takes the database "
            "from one valid state to another, preserving all defined rules and "
            "constraints. Isolation ensures concurrently executing transactions do not "
            "interfere with each other, as if they were executed serially. Durability "
            "ensures that once a transaction is committed, its changes persist even in "
            "the event of a system failure."
        ),
        "rubric": [
            {"id": "p1", "text": "Atomicity means that a transaction executes on an all-or-nothing basis", "marks": 2.5},
            {"id": "p2", "text": "Consistency means that a transaction moves the database from one valid state to another valid state", "marks": 2.5},
            {"id": "p3", "text": "Isolation means that concurrently executing transactions do not interfere with one another", "marks": 2.5},
            {"id": "p4", "text": "Durability means that once a transaction is committed, its changes persist even after a system failure", "marks": 2.5},
        ],
    },

    # -------------------- Data Structures & Algorithms ----------------------
    {
        "qid": "DS01", "subject": "Data Structures & Algorithms",
        "question": "Explain the difference between a stack and a queue, with real-world examples.",
        "max_marks": 5,
        "reference_answer": (
            "A stack follows Last-In-First-Out (LIFO) order, where the most recently "
            "added element is removed first, like a stack of plates. A queue follows "
            "First-In-First-Out (FIFO) order, where the earliest added element is "
            "removed first, like a queue of people waiting in line."
        ),
        "rubric": [
            {"id": "p1", "text": "A stack follows a last-in-first-out order, and a correct real-world example is given", "marks": 2.5,
             "binding": {"subject_terms": ["stack"], "correct_value_terms": ["last-in-first-out", "last in first out", "lifo"],
                          "contradicting_value_terms": ["first-in-first-out", "first in first out", "fifo"]}},
            {"id": "p2", "text": "A queue follows a first-in-first-out order, and a correct real-world example is given", "marks": 2.5,
             "binding": {"subject_terms": ["queue"], "correct_value_terms": ["first-in-first-out", "first in first out", "fifo"],
                          "contradicting_value_terms": ["last-in-first-out", "last in first out", "lifo"]}},
        ],
    },
    {
        "qid": "DS02", "subject": "Data Structures & Algorithms",
        "question": "What is the time complexity of binary search, and why? Explain the algorithm steps.",
        "max_marks": 10,
        "reference_answer": (
            "Binary search has a time complexity of O(log n) because on a sorted array, "
            "it repeatedly compares the target value to the middle element and eliminates "
            "half of the remaining search space on each comparison. The steps are: (1) "
            "find the middle index of the current range, (2) if the middle element equals "
            "the target, return it, (3) if the target is smaller, repeat the search on the "
            "left half, (4) if larger, repeat on the right half, and (5) continue until the "
            "element is found or the range is empty. Because the search space halves each "
            "step, the number of steps needed is proportional to log base 2 of n."
        ),
        "rubric": [
            {"id": "p1", "text": "Binary search has a time complexity of O(log n)", "marks": 2},
            {"id": "p2", "text": "The reasoning given correctly explains that the search space is halved with each comparison", "marks": 3},
            {"id": "p3", "text": "Binary search requires the array to be sorted beforehand", "marks": 2},
            {"id": "p4", "text": "The algorithm steps are correctly described, including finding the middle element, comparing it to the target, and recursing into the left or right half", "marks": 3},
        ],
    },
    {
        "qid": "DS03", "subject": "Data Structures & Algorithms",
        "question": "Explain how a hash table handles collisions.",
        "max_marks": 10,
        "reference_answer": (
            "A collision occurs when two different keys hash to the same index in the "
            "underlying array. Chaining resolves this by storing a linked list (or "
            "similar structure) of all key-value pairs that hash to the same index, so "
            "multiple entries can coexist at one slot. Open addressing resolves it "
            "instead by probing for the next available slot within the array itself, "
            "using strategies such as linear probing, quadratic probing, or double "
            "hashing, until an empty slot is found."
        ),
        "rubric": [
            {"id": "p1", "text": "A hash collision is correctly defined as two different keys hashing to the same index", "marks": 2},
            {"id": "p2", "text": "Chaining is explained as storing a linked list of key-value pairs at the same index", "marks": 4},
            {"id": "p3", "text": "Open addressing is explained as probing for the next available slot within the array itself", "marks": 4},
        ],
    },
]

# ---------------------------------------------------------------------------
# 2. STUDENT ANSWER VARIANTS PER QUESTION (4 quality tiers)
#    Each tier explicitly lists which rubric point IDs it genuinely covers.
#    human_score = sum(marks of covered points) + small human-grader noise.
# ---------------------------------------------------------------------------

ANSWER_VARIANTS = {
    "PY01": {
        "excellent": ("Lists are mutable so you can add, remove or change elements after "
                      "creation, while tuples are immutable and cannot be changed once made. "
                      "Lists use square brackets [] and tuples use parentheses (). Since "
                      "tuples can't change, they're hashable and can be used as dictionary "
                      "keys, which lists cannot.", ["p1", "p2", "p3"]),
        "good": ("A list can be changed after it is created, but a tuple cannot be "
                 "modified once it is defined. Lists use [] and tuples use ().", ["p1", "p2"]),
        "weak": ("Lists and tuples are both used to store multiple values in Python. "
                 "Lists use square brackets.", ["p2"]),
        "poor": ("A list is a data type in Python used to store data. It is very useful "
                 "for programming.", []),
    },
    "PY02": {
        "excellent": ("A generator is a special kind of function that uses 'yield' instead "
                      "of 'return' to produce values one at a time, pausing and remembering "
                      "its state between each call. Because it never holds the whole "
                      "sequence in memory at once, it is far more memory-efficient than "
                      "building a full list.", ["p1", "p2", "p3"]),
        "good": ("A generator uses yield to return values one by one instead of all at "
                 "once, which saves memory compared to a list.", ["p1", "p3"]),
        "weak": ("A generator is a function in Python that generates values.", []),
        "poor": ("Generators are used in Python for loops and iteration over data.", []),
    },
    "PY03": {
        "excellent": ("NumPy broadcasting lets you do element-wise operations on arrays of "
                      "different shapes by stretching the smaller array across the larger "
                      "one without copying data. Two dimensions are compatible when they "
                      "are equal or one of them is 1. For instance, adding a 1D array of "
                      "shape (3,) to a 2D array of shape (3,3) adds the vector to every row "
                      "of the matrix. This avoids writing explicit for-loops and is much "
                      "faster.", ["p1", "p2", "p3", "p4"]),
        "good": ("Broadcasting allows NumPy to perform operations between arrays of "
                 "different shapes by expanding the smaller array. For example, a (3,) "
                 "array added to a (3,3) array gets added to every row.", ["p1", "p3"]),
        "weak": ("Broadcasting in NumPy is used to add two arrays together even if they "
                 "are different sizes.", ["p1"]),
        "poor": ("NumPy is a Python library for numerical computation with arrays and "
                 "matrices.", []),
    },
    "ST01": {
        "excellent": ("The Central Limit Theorem says that as sample size increases, the "
                      "sampling distribution of the sample mean approaches a normal "
                      "distribution, no matter what the shape of the original population "
                      "is, as long as the population variance is finite. This usually "
                      "kicks in around a sample size of 30. It matters because it lets us "
                      "use normal-based confidence intervals and hypothesis tests even on "
                      "non-normal data.", ["p1", "p2", "p3", "p4"]),
        "good": ("CLT states that the distribution of sample means becomes approximately "
                 "normal as sample size grows large, even if the population isn't normal. "
                 "A sample size of around 30 is often considered large enough.", ["p1", "p2", "p3"]),
        "weak": ("The Central Limit Theorem is about sample means becoming normal.", ["p1"]),
        "poor": ("Statistics involves collecting and analyzing numerical data for "
                 "decision making.", []),
    },
    "ST02": {
        "excellent": ("A Type I error happens when we reject a null hypothesis that is "
                      "actually true - a false positive - and its probability is alpha. A "
                      "Type II error happens when we fail to reject a null hypothesis that "
                      "is actually false - a false negative - with probability beta.", ["p1", "p2"]),
        "good": ("Type I error is rejecting a true null hypothesis. Type II error is not "
                 "rejecting a false null hypothesis.", ["p1", "p2"]),
        "weak": ("Type I and Type II errors are mistakes that can happen during "
                 "hypothesis testing when we make wrong conclusions.", []),
        "poor": ("Hypothesis testing is used to test claims about a population using "
                 "sample data.", []),
    },
    "ST03": {
        "excellent": ("Correlation just measures how strongly two variables move together "
                      "statistically, but it doesn't mean one causes the other. Causation "
                      "means one variable directly produces a change in the other. For "
                      "example, ice cream sales and drowning deaths are correlated because "
                      "both rise in summer, but the real cause behind both is hot weather, "
                      "not one causing the other.", ["p1", "p2", "p3"]),
        "good": ("Correlation means two variables are related, but causation means one "
                 "actually causes the other. For example, more ice cream sales don't cause "
                 "more drownings even though they're correlated.", ["p1", "p2"]),
        "weak": ("Correlation and causation are different things in statistics.", []),
        "poor": ("Data analysis helps identify patterns and relationships in datasets for "
                 "business use.", []),
    },
    "ML01": {
        "excellent": ("Bias is the error from oversimplifying the real relationship with a "
                      "model that's too simple, and it leads to underfitting. Variance is "
                      "the error from a model being too sensitive to fluctuations in the "
                      "training set, leading to overfitting. There is a tradeoff because "
                      "making a model more complex to reduce bias usually increases "
                      "variance, and vice versa, so we aim to pick the complexity that "
                      "minimizes total error, which combines bias squared, variance and "
                      "irreducible error.", ["p1", "p2", "p3", "p4"]),
        "good": ("High bias means the model is too simple and underfits, while high "
                 "variance means the model is too complex and overfits. There's a tradeoff "
                 "between the two when choosing model complexity.", ["p1", "p2", "p3"]),
        "weak": ("Bias and variance are two types of errors a machine learning model can "
                 "have.", []),
        "poor": ("Machine learning models are trained on data to make predictions on new "
                 "data.", []),
    },
    "ML02": {
        "excellent": ("Overfitting happens when a model memorizes noise and specific "
                      "details of the training data, so it performs very well on training "
                      "data but poorly on new, unseen data. Regularization fixes this by "
                      "adding a penalty term to the loss function that discourages large "
                      "coefficients or overly complex models. L1 (Lasso) can shrink some "
                      "coefficients all the way to zero, effectively doing feature "
                      "selection, while L2 (Ridge) shrinks coefficients smoothly but rarely "
                      "to exactly zero. Both trade a little extra bias for a large "
                      "reduction in variance.", ["p1", "p2", "p3", "p4"]),
        "good": ("Overfitting is when a model fits the training data too closely and "
                 "doesn't generalize well. Regularization adds a penalty to the loss "
                 "function to keep the model simpler and reduce overfitting.", ["p1", "p2"]),
        "weak": ("Overfitting is a common problem in machine learning where the model "
                 "performs badly.", []),
        "poor": ("Machine learning has supervised and unsupervised learning types used in "
                 "many applications.", []),
    },
    "ML03": {
        "excellent": ("Bagging trains several models in parallel, each on a different "
                      "bootstrap sample of the data, and then averages or votes on their "
                      "predictions; this mainly reduces variance, and Random Forest is the "
                      "classic example. Boosting instead trains models one after another, "
                      "where each new model tries to fix the mistakes of the previous ones "
                      "by giving more weight to misclassified examples, which mainly "
                      "reduces bias; AdaBoost and Gradient Boosting are examples of this.", ["p1", "p2", "p3"]),
        "good": ("Bagging trains multiple models in parallel on random samples and "
                 "combines them, like Random Forest. Boosting trains models sequentially, "
                 "where each one corrects the errors of the last one.", ["p1", "p2"]),
        "weak": ("Bagging and boosting are ensemble techniques used to combine multiple "
                 "models in machine learning.", []),
        "poor": ("Decision trees are a popular machine learning algorithm used for "
                 "classification tasks.", []),
    },
    "DB01": {
        "excellent": ("Normalization organizes tables in a relational database to cut down "
                      "on redundant data and prevent update, insert and delete anomalies. "
                      "1NF requires every attribute to hold only atomic values with no "
                      "repeating groups. 2NF builds on 1NF by requiring every non-key "
                      "attribute to depend on the whole primary key, removing partial "
                      "dependency. 3NF builds on 2NF by requiring that no non-key attribute "
                      "depends on another non-key attribute, removing transitive "
                      "dependency.", ["p1", "p2", "p3", "p4"]),
        "good": ("Normalization reduces redundancy in database tables. 1NF means atomic "
                 "values only, 2NF removes partial dependency on the key, and 3NF removes "
                 "transitive dependency.", ["p2", "p3", "p4"]),
        "weak": ("Normalization is used in DBMS to organize data properly in tables.", ["p1"]),
        "poor": ("A database is a collection of tables used to store information for "
                 "applications.", []),
    },
    "DB02": {
        "excellent": ("A JOIN combines rows from two or more tables based on a related "
                      "column between them. An INNER JOIN returns only rows that have "
                      "matching values in both tables. A LEFT JOIN returns every row from "
                      "the left table plus matching rows from the right table, putting "
                      "NULL where there's no match. For example, joining Students with "
                      "Enrollments using INNER JOIN only shows students who enrolled in at "
                      "least one course, while LEFT JOIN also shows students with zero "
                      "enrollments.", ["p1", "p2", "p3", "p4"]),
        "good": ("A JOIN is used to combine data from two tables based on a common "
                 "column. INNER JOIN returns only matching rows, while LEFT JOIN returns "
                 "all rows from the left table even without a match.", ["p1", "p2", "p3"]),
        "weak": ("JOIN is an SQL keyword used to combine tables together in a query.", ["p1"]),
        "poor": ("SQL is a language used to manage and query relational databases.", []),
    },
    "DB03": {
        "excellent": ("ACID stands for Atomicity, Consistency, Isolation and Durability. "
                      "Atomicity means a transaction either completes fully or not at all. "
                      "Consistency means a transaction moves the database from one valid "
                      "state to another without breaking any rules. Isolation means "
                      "concurrent transactions don't affect each other as if they ran one "
                      "after another. Durability means once a transaction is committed, its "
                      "effects survive even a system crash.", ["p1", "p2", "p3", "p4"]),
        "good": ("ACID properties ensure reliable transactions. Atomicity means all-or-"
                 "nothing execution, and durability means committed changes are permanent "
                 "even after a crash.", ["p1", "p4"]),
        "weak": ("ACID is important for database transactions to work correctly.", []),
        "poor": ("Transactions in a database are operations performed on data stored in "
                 "tables.", []),
    },
    "DS01": {
        "excellent": ("A stack works on Last-In-First-Out order, so the last element added "
                      "is the first one removed, like a stack of plates where you take from "
                      "the top. A queue works on First-In-First-Out order, so the first "
                      "element added is the first one removed, like people standing in a "
                      "line waiting for a bus.", ["p1", "p2"]),
        "good": ("A stack is LIFO, last in first out, and a queue is FIFO, first in first "
                 "out.", ["p1", "p2"]),
        "weak": ("Stacks and queues are both linear data structures used to store "
                 "elements.", []),
        "poor": ("Data structures are used in programming to organize and store data "
                 "efficiently.", []),
    },
    "DS02": {
        "excellent": ("Binary search runs in O(log n) time because each comparison cuts "
                      "the remaining search space in half. The steps are: find the middle "
                      "element of the current range; if it equals the target, return it; if "
                      "the target is smaller, search the left half; if larger, search the "
                      "right half; repeat until found or the range becomes empty. It only "
                      "works on a sorted array, and because the space keeps halving, the "
                      "number of steps grows proportional to log base 2 of n.", ["p1", "p2", "p3", "p4"]),
        "good": ("Binary search is O(log n) because it keeps dividing the search space in "
                 "half. It checks the middle element and then searches the left or right "
                 "half depending on whether the target is smaller or larger. It requires a "
                 "sorted array.", ["p1", "p2", "p3"]),
        "weak": ("Binary search is faster than linear search because it uses a divide "
                 "approach.", ["p1"]),
        "poor": ("Searching algorithms are used to find elements in a list or array of "
                 "data.", []),
    },
    "DS03": {
        "excellent": ("A collision happens when two different keys map to the same index "
                      "in a hash table's underlying array. Chaining handles this by keeping "
                      "a linked list of all key-value pairs that land on the same index, so "
                      "several entries can live at one slot. Open addressing instead looks "
                      "for the next free slot inside the array itself, using approaches "
                      "like linear probing, quadratic probing or double hashing, until it "
                      "finds an empty one.", ["p1", "p2", "p3"]),
        "good": ("A collision is when two keys hash to the same slot. Chaining stores "
                 "multiple items at the same index using a linked list. Open addressing "
                 "finds another empty slot in the array instead.", ["p1", "p2", "p3"]),
        "weak": ("Hash tables can have collisions when two keys map to the same location, "
                 "which needs to be handled properly.", ["p1"]),
        "poor": ("Hash tables are used to store key-value pairs for fast lookup in "
                 "programming.", []),
    },
}

TIER_NOISE = {"excellent": (-0.5, 0.5), "good": (-0.5, 0.5),
              "weak": (-0.5, 0.5), "poor": (-0.3, 0.3)}

# ---------------------------------------------------------------------------
# 2b. ADVERSARIAL TEST-ONLY CASES.
#     A pure keyword/TF-IDF baseline is known to fail in two directions:
#       (a) a CORRECT answer phrased with different vocabulary (paraphrase)
#           gets under-scored because there is little lexical overlap;
#       (b) an INCORRECT/incoherent answer that recycles the reference
#           answer's vocabulary (keyword stuffing) gets over-scored.
#     These are injected as held-out TEST examples (never used to fit the
#     model) specifically to surface this limitation in the Week-2 error
#     analysis and motivate the Week-3/4 embedding-based "improved
#     approach" comparison required by Section 6.
# ---------------------------------------------------------------------------
EXTRA_TEST_CASES = [
    {
        "qid": "PY01", "tier": "paraphrase_trap", "human_score": 2.5,
        # Only the mutable/immutable idea (p1, 2 marks) is actually present;
        # brackets syntax (p2) and hashability (p3) are not mentioned.
        "covered": ["p1"],
        "student_answer": (
            "One of these two collection types can have its contents "
            "rearranged after you build it, whereas the other is locked in "
            "place the moment you create it, so its items can never later "
            "be reordered or changed."
        ),
    },
    {
        "qid": "ML02", "tier": "keyword_stuffing", "human_score": 1.5,
        "covered": [],
        "student_answer": (
            "Overfitting overfitting regularization L1 L2 loss function "
            "coefficients training data generalization penalty term model "
            "complexity Lasso Ridge."
        ),
    },
    {
        "qid": "ST01", "tier": "paraphrase_trap", "human_score": 9.5,
        # All four CLT concepts are genuinely present, just in casual
        # language: normality of the sample-mean distribution (p1), holds
        # regardless of population shape (p2), n ~ 30 rule of thumb (p3),
        # and why it matters for inference (p4).
        "covered": ["p1", "p2", "p3", "p4"],
        "student_answer": (
            "No matter how strangely shaped your original data is, if you "
            "keep drawing bigger and bigger groups of samples and "
            "averaging each group, those averages start clustering into a "
            "bell-shaped pattern once each group has roughly thirty or "
            "more items. That is basically why we are allowed to use "
            "bell-curve-based confidence math even on messy, non-normal "
            "real-world data."
        ),
    },
    {
        "qid": "DB02", "tier": "keyword_stuffing", "human_score": 1.0,
        "covered": [],
        "student_answer": (
            "JOIN INNER JOIN LEFT JOIN tables rows columns match SQL "
            "database query example students enrollments records."
        ),
    },
    {
        "qid": "DS02", "tier": "paraphrase_trap", "human_score": 9.0,
        # All four binary-search concepts are genuinely present in
        # different words: O(log n) (p1), halving reasoning (p2), sorted
        # precondition (p3), and the mid/compare/recurse steps (p4).
        "covered": ["p1", "p2", "p3", "p4"],
        "student_answer": (
            "Because you cut the problem in half every single time you "
            "check the middle item, the number of checks needed grows "
            "extremely slowly even for huge lists - proportional to log "
            "base two of the list size. The list has to be sorted first; "
            "then you just keep checking the center element and throwing "
            "away whichever side clearly cannot contain what you are "
            "looking for, until you find it or run out of items."
        ),
    },

    # ---- Week 5: robustness/failure-mode cases beyond paraphrase & ----
    # ---- keyword-stuffing - blank, off-topic, refusal, confidently ----
    # ---- wrong (inverted facts), and partial-with-an-embedded-error ----
    {
        "qid": "ST03", "tier": "blank", "human_score": 0.0,
        "covered": [],
        "student_answer": "N/A",
    },
    {
        "qid": "DB03", "tier": "off_topic", "human_score": 0.0,
        "covered": [],
        "student_answer": (
            "Our college cricket team won the inter-department tournament "
            "last week after a tense final over. The captain scored a "
            "brilliant half-century and the whole team celebrated together "
            "in the canteen afterwards."
        ),
    },
    {
        "qid": "ST02", "tier": "refusal", "human_score": 0.0,
        "covered": [],
        "student_answer": (
            "I don't know the answer to this question. I did not get a "
            "chance to study this topic before the exam."
        ),
    },
    {
        "qid": "DS01", "tier": "confidently_wrong", "human_score": 0.5,
        # Fluent, uses the right vocabulary and the right examples, but
        # SWAPS the two definitions - LIFO/FIFO are attributed backwards.
        # Tests whether the grader rewards vocabulary overlap even when
        # the actual claim is factually inverted.
        "covered": [],
        "student_answer": (
            "A stack follows a First-In-First-Out order, similar to people "
            "standing in a line waiting for a bus, while a queue follows a "
            "Last-In-First-Out order, similar to a stack of plates where "
            "you take the top one first."
        ),
    },
    {
        "qid": "ML01", "tier": "mixed_with_error", "human_score": 5.0,
        # Bias and variance are explained correctly (p1, p2) but the
        # tradeoff claim is factually backwards (asserts they co-occur
        # rather than trade off), so p3 should NOT be credited.
        "covered": ["p1", "p2"],
        "student_answer": (
            "Bias is the error from using a model that is too simple, "
            "which causes underfitting. Variance is the error from a "
            "model being too sensitive to the training data, which causes "
            "overfitting. The tradeoff means that high bias and high "
            "variance always happen together in every model, so you "
            "cannot reduce one without the other also getting worse."
        ),
    },
]


def score_for(points_covered, rubric, max_marks, tier):
    marks_map = {p["id"]: p["marks"] for p in rubric}
    base = sum(marks_map[pid] for pid in points_covered)
    lo, hi = TIER_NOISE[tier]
    noisy = base + random.uniform(lo, hi)
    return round(max(0, min(max_marks, noisy)) * 2) / 2  # round to nearest 0.5


def build_dataset():
    rows = []
    sid_counter = 1000
    for q in QUESTIONS:
        variants = ANSWER_VARIANTS[q["qid"]]
        for tier, (answer_text, covered) in variants.items():
            sid_counter += 1
            score = score_for(covered, q["rubric"], q["max_marks"], tier)
            rows.append({
                "qid": q["qid"],
                "subject": q["subject"],
                "question": q["question"],
                "reference_answer": q["reference_answer"],
                "rubric": json.dumps(q["rubric"]),
                "max_marks": q["max_marks"],
                "student_id": f"S{sid_counter}",
                "student_answer": answer_text,
                "points_covered": json.dumps(covered),
                "human_score": score,
                "quality_tier": tier,
            })
    return rows


def add_split(rows, train_frac=0.7, seed=42):
    rnd = random.Random(seed)
    by_qid = {}
    for r in rows:
        by_qid.setdefault(r["qid"], []).append(r)
    for qid, group in by_qid.items():
        idx = list(range(len(group)))
        rnd.shuffle(idx)
        n_train = max(1, round(len(group) * train_frac))
        for i, orig_i in enumerate(idx):
            group[orig_i]["split"] = "train" if i < n_train else "test"
    return rows


def add_validation_split(rows, val_frac=0.25, seed=43):
    """
    Week 5: carves a validation split OUT OF the existing 'train' rows only
    - the 'test' split (used for every Week 2-4 headline number) is never
    touched, so all previously reported metrics remain reproducible. Done
    at the whole-pool level (not per-question, unlike add_split) because
    each question only contributes 3 train rows, too few to subdivide
    per-question; the validation set is used only to tune a single global
    decision threshold, which doesn't need per-question balance.
    """
    rnd = random.Random(seed)
    train_rows = [r for r in rows if r["split"] == "train"]
    idx = list(range(len(train_rows)))
    rnd.shuffle(idx)
    n_val = round(len(train_rows) * val_frac)
    val_idx = set(idx[:n_val])
    for i, r in enumerate(train_rows):
        if i in val_idx:
            r["split"] = "validation"
    return rows


def build_extra_test_rows():
    q_by_id = {q["qid"]: q for q in QUESTIONS}
    rows = []
    sid_counter = 2000
    for case in EXTRA_TEST_CASES:
        q = q_by_id[case["qid"]]
        sid_counter += 1
        rows.append({
            "qid": q["qid"], "subject": q["subject"], "question": q["question"],
            "reference_answer": q["reference_answer"], "rubric": json.dumps(q["rubric"]),
            "max_marks": q["max_marks"], "student_id": f"S{sid_counter}",
            "student_answer": case["student_answer"], "points_covered": json.dumps(case["covered"]),
            "human_score": case["human_score"], "quality_tier": case["tier"],
            "split": "test",
        })
    return rows


if __name__ == "__main__":
    rows = build_dataset()
    rows = add_split(rows)
    rows = add_validation_split(rows)
    rows = rows + build_extra_test_rows()

    fieldnames = ["qid", "subject", "question", "reference_answer", "rubric",
                  "max_marks", "student_id", "student_answer", "points_covered",
                  "human_score", "quality_tier", "split"]

    out_path = OUT_DIR / "exam_grading_dataset.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "validation")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(f"Generated {len(rows)} records -> {out_path}")
    print(f"  Subjects: {sorted(set(r['subject'] for r in rows))}")
    print(f"  Train: {n_train}  Validation: {n_val}  Test: {n_test}")
