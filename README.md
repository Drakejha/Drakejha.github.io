Drake Jha
Department of Atmospheric and Oceanic Sciences, UCLA
AOS C111 / C204: Introduction to Machine Learning for the Physical Sciences
Dr. Alexander Lozinski
December 5, 2025

Code:
[Open the project notebook in Google Colab](https://colab.research.google.com/github/Drakejha/Drakejha.github.io/blob/main/Drake_Jha_Final_Project (1).ipynb)

Introduction:

In this project I use machine learning to predict the strength of a poker hand from the five cards that are dealt. The goal is to look at the suits and ranks of the cards and classify the hand as “nothing”, “one pair”, “two pairs”, “full house”, all the way up to “royal flush”. This is a good test problem because there are many possible hands, but strong hands are rare, so the classes are very unbalanced and harder to learn.

The dataset I use is the Poker Hand dataset (from the course zip file, originally from the UCI Machine Learning Repository). Each row has ten integer features: five suits (S1–S5) and five ranks (C1–C5). The label, called *hand*, is an integer from 0 to 9 that tells us what type of poker hand it is. My machine-learning task is to map these ten input numbers to the correct hand class.

To solve this as a supervised classification problem, I built a simple baseline model and then three real models: Logistic Regression, a Decision Tree, and a Random Forest. I trained the models on part of the data, used a validation split for tuning, and finally evaluated everything on a large 1,000,000-row test set. To judge model performance I used error metrics from Lecture 2.3 for classification: accuracy, confusion matrices, and precision, recall, and F1-score for each hand class. Overall, the Random Forest gave the best accuracy and handled strong hands better than the other models, although all models still struggled on the rarest hand types.

Data:

For this project I used the Poker Hand dataset from the course zip file (originally from the UCI Machine Learning Repository). Each row in the dataset is one 5-card poker hand, dealt from a standard 52-card deck. The goal is to predict what type of hand it is.

The dataset has 11 columns. Ten of them are the input features: S1–S5 and C1–C5, which are the suit and rank (card value) of the five cards, all stored as integers. The last column is “hand”, which is the target label from 0 to 9. These labels represent the hand types: 0 = nothing in hand (no pair), 1 = one pair, 2 = two pairs, 3 = three of a kind, 4 = straight, 5 = flush, 6 = full house, 7 = four of a kind, 8 = straight flush, and 9 = royal flush. So the models take the 10 numeric card columns as input and try to predict the integer hand class.

The zip file already comes with a fixed split: 25,010 hands for training and 1,000,000 hands for testing, each with 11 columns. In my notebook I loaded these files using pandas (read_csv) and assigned the column names listed above.

One important property of this dataset is that the classes are very unbalanced. When I counted the number of examples of each hand class in the training set, I found that most hands are class 0 or 1 (no pair or one pair), which together make up over 90% of the data. The very strong hands, like full houses, four of a kind, straight flushes, and royal flushes (classes 6 to 9), are extremely rare. This class imbalance matters because a model can get a pretty good overall accuracy just by focusing on the common classes, while still performing poorly on the rare but interesting strong hands.

The pre-processing for this project was kept simple and stayed within the material from the course. I read the CSV files with pandas, set the column names, used only the numeric card columns S1–S5 and C1–C5 as features, and used “hand” as the target. I checked that the data types were integers and that there were no missing values. I did not use any scaling or advanced feature engineering, because the models used (logistic regression, decision tree, and random forest) can work directly with these integer-coded inputs.

Modeling:

In this project I treated poker-hand recognition as a supervised multi-class classification problem. Each hand is described by 10 numeric input features (the suits and ranks of the 5 cards), and the target is one of 10 hand classes (0 = “nothing in hand” up to 9 = “royal flush”). Because the dataset is very imbalanced (most hands are class 0 or 1), I first built a simple baseline model that always predicts the most common class (class 0). This baseline gets about 0.50 test accuracy, so any real model has to beat 50% accuracy to be useful.

To choose and tune my models, I did an internal train/validation split of the 25,010 training hands using train_test_split(..., stratify=y), keeping 80% for training and 20% for validation. For each model I measured performance on the validation set using the main error metrics from Lecture 2.3 for classification: overall accuracy, the classification report (precision, recall, and F1-score for each class), and the confusion matrix. These tools help me see not only how often the model is right overall, but also how well it handles the rare but important strong hands.

I stayed within the algorithms covered in the course: logistic regression, a decision tree, and a random forest. Logistic regression is a linear classifier and reached about 0.50 accuracy, only slightly better than the baseline, and it mainly predicts the common classes. The single decision tree did similarly and tended to overfit. The random forest (an ensemble of many decision trees) clearly worked best: on the validation set it reached about 0.61 accuracy, beating the other models and doing better on several non-trivial hand classes, although it still struggles with the very rare hands like straight flush and royal flush. After choosing the random forest as my final model, I retrained all models on the full training set and reported their test accuracies on the separate 1,000,000-hand test file.

Results:

Figure 1:

<img width="713" height="388" alt="image" src="https://github.com/user-attachments/assets/537322e5-0f73-4a5f-bbb7-f66de4a3dc8f" />

Figure 2:

<img width="533" height="455" alt="image" src="https://github.com/user-attachments/assets/0f2d77ac-e56e-4fa0-8d77-c184c5d2894b" />

Figure 3:

<img width="588" height="390" alt="image" src="https://github.com/user-attachments/assets/f6bc7a49-d518-4894-afa1-4eb4656d4fd5" />

Figure 1 shows the class distribution in the training data. Most hands are class 0 (“nothing in hand”) or class 1 (“one pair”), while the stronger hands, especially classes 6–9 (full house, four of a kind, straight flush, royal flush), have only a few examples each. This confirms that the dataset is extremely imbalanced. Because of this, a model can get a pretty high overall accuracy just by predicting the common classes, so I have to be careful when interpreting accuracy.

Figure 2 compares the test accuracies of the four models on the 1,000,000-hand test set. The baseline model (always predicting class 0) and logistic regression both reach about 0.50 accuracy, which is only slightly better than random guessing among 10 classes but already shows the effect of class imbalance. The decision tree does a bit worse overall, around 0.48, and seems to overfit the training data. The random forest clearly performs the best, reaching about 0.61 test accuracy, which is roughly 10 percentage points higher than the baseline and the other models.

Figure 3 shows the confusion matrix for the random forest on the test set. The diagonal entries for classes 0 and 1 are very large, meaning the model is very good at recognizing “nothing in hand” and “one pair.” For the rare strong hands (classes 5–9), the diagonal values are much smaller and there is more mass in the first few columns, so those hands are often misclassified as weaker ones. This matches the precision and recall patterns from the classification report: the random forest improves overall accuracy and does better on some minority classes, but it still struggles with the very rare hands because there just are not many training examples for them.

Discussion:

Overall, the results fit what was expected from the class imbalance and from the types of models I used. Because most examples are “nothing in hand” or “one pair,” even a simple baseline that always predicts class 0 already reaches about 0.50 accuracy. Logistic regression ends up very close to this baseline. It is a linear model, so it mostly learns to separate the common classes and has trouble modeling the more complicated patterns needed to recognize rare strong hands.

The decision tree can, in theory, learn more complex decision boundaries, but on this very imbalanced dataset it tends to overfit the training data. That is why its test accuracy (about 0.48) is actually a bit worse. The confusion matrix and classification report show that the tree does not treat the rare classes much better than the simpler models.

The random forest performs best, with test accuracy around 0.61. This makes sense because a random forest is an ensemble of many decision trees trained on different subsets of the data, so averaging their predictions reduces variance and usually generalizes better. In the confusion matrix, the random forest is better at separating “nothing,” “one pair,” and “two pair” than the other models, but it still has low recall for the very strongest hands (classes 7–9). This matches the ideas from Lecture 2.3: even when overall accuracy improves, there's still a need to check per-class precision and recall, especially for minority classes. In this project, accuracy alone would hide the fact that all models, including the random forest, still miss most of the rare but most valuable poker hands.

Conclusion:
In this project I built and compared several machine-learning models to predict the strength of poker hands. A simple baseline model that always predicts “no made hand” (class 0) already reaches about 50% test accuracy, because more than half of all examples are in that class. This showed me that accuracy by itself can be misleading when the dataset is very imbalanced, and that every “real” model should first be compared against a basic baseline.

Among the real models, the Random Forest classifier performed the best. On the held-out test set it achieved an accuracy of about 0.61, which is clearly higher than Logistic Regression (around 0.50) and the single Decision Tree (around 0.48). The confusion matrix also showed that all models, including the Random Forest, are much better at predicting common hands than rare ones. The strongest hands (like full house, four-of-a-kind, straight flush, and royal flush) still get mixed up with weaker classes because there are so few examples of them.

If I continued this project, I would focus on improving performance for these rare, high-value hands. Possible next steps would be to handle the class imbalance more directly (for example by oversampling rare classes or using class-weighted training) and to add more poker-style features, such as “number of cards with the same rank” or “is_flush.” Overall, this project showed that machine learning can learn meaningful patterns from poker data, but it also highlighted how important data balance and good evaluation metrics are when judging a model.

References:

Cattral, Robert, and Franz Oppacher. Poker Hand. UCI Machine Learning Repository, 2006, https://doi.org/10.24432/C5KW38
.

Géron, Aurélien. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow. 2nd ed., O’Reilly Media, 2019.

Pedregosa, Fabian, et al. “Scikit-learn: Machine Learning in Python.” Journal of Machine Learning Research, vol. 12, 2011, pp. 2825–2830.

“API Reference.” scikit-learn, scikit-learn.org/stable/modules/classes.html.

“Poker Hand Rankings.” World Series of Poker, www.wsop.com/poker-hands/
.



