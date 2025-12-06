Introduction:

In this project I use machine learning to predict the strength of a poker hand from the five cards that are dealt. The idea is simple: if a computer can look at the suits and ranks of the cards and guess whether the hand is “nothing”, “one pair”, “two pairs”, “full house”, etc., then it has learned something real about how poker works. Poker is a good test problem because there are many possible hands, but strong hands are rare, which makes the classification task unbalanced and challenging.

The dataset I chose contains tens of thousands of example poker hands for training, and one million hands for testing. Each hand is described by ten integers: five suits (S1–S5) and five card ranks (C1–C5). The target label, called hand, is a number from 0 to 9 that encodes the type of poker hand (0 = “nothing in hand”, 1 = “one pair”, up to 9 = “royal flush”). My main goal is to train models that can map these ten input numbers to the correct hand class.

To solve this as a supervised learning problem, I applied several classification algorithms we studied in class. I first built a very simple baseline that always predicts the most common class in the training data, so I can see whether my real models are actually doing better than a “dumb” guesser. Then I trained three models: Logistic Regression, a Decision Tree, and a Random Forest. All of them were trained on part of the data and tuned/evaluated on a separate validation split, before being tested on the large 1,000,000-row test set.

To measure how good each model is, I used the error metrics from Lecture 2.3 for classification: accuracy, confusion matrices, and the precision, recall, and F1-scores for each class. Overall accuracy tells me how often the model is correct, but the confusion matrix and per-class scores show a more detailed picture, especially for the rare strong hands. In the results, I find that the Random Forest clearly performs best on overall accuracy and does a better job than the other models at picking up stronger hands, although it still struggles with the very rare classes.

Data

For this project I used the Poker Hand dataset (provided in the course zip file, originally from the UCI Machine Learning Repository).
Each row in the dataset represents one 5-card poker hand, dealt from a standard 52-card deck, and the goal is to predict what type of hand it is.

Features and target

The dataset has 11 columns:

S1, C1 – suit and rank (card value) of card 1

S2, C2 – suit and rank of card 2

S3, C3 – suit and rank of card 3

S4, C4 – suit and rank of card 4

S5, C5 – suit and rank of card 5

hand – target label (0–9), which tells us what kind of poker hand it is

The hand label uses the following integer codes:

0 = “Nothing in hand” (no pair)

1 = “One pair”

2 = “Two pairs”

3 = “Three of a kind”

4 = “Straight”

5 = “Flush”

6 = “Full house”

7 = “Four of a kind”

8 = “Straight flush”

9 = “Royal flush”

So the input features are the 10 numeric columns S1–S5 and C1–C5, and the output we want to predict is the integer hand class.

Train / test split provided in the file

The zip file already comes with a predefined split:

Training set: 25,010 hands, 11 columns

Test set: 1,000,000 hands, 11 columns

In my notebook I loaded:

train_df = pd.read_csv(train_path, header=None, names=col_names)
test_df  = pd.read_csv(test_path,  header=None, names=col_names)


and checked the shapes and column types to confirm that everything was numeric and there were no missing values.

Class imbalance

One important property of this dataset is that the classes are very unbalanced.
When I counted how many examples there are of each hand class in the training set, I got:

Class 0 (nothing): 12,493 examples (~50%)

Class 1 (one pair): 10,599 examples (~42%)

Class 2 (two pairs): 1,206 examples (~4.8%)

Class 3 (three of a kind): 513 examples (~2.1%)

Class 4 (straight): 93 examples (~0.4%)

Class 5 (flush): 54 examples (~0.2%)

Class 6 (full house): 36 examples (~0.1%)

Class 7 (four of a kind): 6 examples

Class 8 (straight flush): 5 examples

Class 9 (royal flush): 5 examples

So about 92% of all hands are just class 0 or 1 (no pair or one pair), and the strongest hands (classes 6–9) are extremely rare.

To visualize this, I plotted a simple bar chart of the class counts:

class_counts = train_df["hand"].value_counts().sort_index()
class_counts.plot(kind="bar")
plt.xlabel("Hand class (0–9)")
plt.ylabel("Count")
plt.title("Class distribution in poker-hand training data")
plt.show()


The plot clearly shows that most examples are in classes 0 and 1, with only tiny bars for the higher classes.
This class imbalance is important, because a model can get a pretty high overall accuracy just by guessing the common classes, while doing badly on the rare but interesting hands (like full houses and royal flushes). I keep this in mind later when I look at the confusion matrix and the precision/recall numbers for each class.

Pre-processing steps

The preprocessing was kept simple and stayed inside the material from the course:

Read the CSV files with pandas and set column names.

Use only numeric features (S1–S5, C1–C5) as inputs and hand as the target.

Confirm that the data types are integers and there are no missing values.

No scaling or advanced feature engineering was used, because the models we applied (logistic regression, decision tree, random forest) can handle integer-coded categories directly for this project.

Modelling

In this project I treated poker-hand recognition as a supervised multi-class classification problem.
Each hand is described by 10 numeric input features (the suits and ranks of the 5 cards), and the goal is to predict one of 10 classes (hand = 0,…,9), which represent different poker hand types.

Because the classes are very imbalanced (most hands are class 0 or 1), I started with a simple baseline model before using any “real” machine learning. The baseline always predicts the most common class in the training data (class 0, “nothing in hand”). This gives a test accuracy of about 0.50, which means that any useful model has to beat 50% accuracy, not just 10% (which is what you’d get by randomly guessing among 10 classes).

Train/validation split

To choose and tune the models, I did an internal split of the 25,010 training examples provided in the file:

80% used for training

20% kept aside as a validation set

I used train_test_split from sklearn.model_selection with stratify=y so that the class proportions in the validation set match the original class imbalance. For each model I trained on the 80% training part, then evaluated performance on the 20% validation part using:

Accuracy

Classification report (precision, recall, f1-score for each class)

Confusion matrix (for the tree-based models)

These evaluation tools come from the error metrics lecture (Lecture 2.3): accuracy, precision, recall, f1, and confusion matrices for classification.

Models used

I stayed within the algorithms covered in the course modules: logistic regression, decision trees, and random forests.

Logistic Regression (multi-class)

I used LogisticRegression from sklearn.linear_model with a higher max_iter to make sure it converged.

This model is a linear classifier: it tries to draw linear decision boundaries in the 10-dimensional feature space to separate the different hand classes.

On the validation set, logistic regression reached an accuracy of about 0.50, which is only slightly better than the baseline. It mainly predicts the common classes (0 and 1) and struggles with rare strong hands.

Decision Tree

Next I used DecisionTreeClassifier from sklearn.tree.

A decision tree splits the data based on feature conditions like “is C3 ≥ 10?” and follows these splits down a tree until it makes a prediction.

On the validation set the decision tree got an accuracy of about 0.49. It did slightly better than logistic regression on some minority classes but was still limited, and it tended to overfit the training data (this is consistent with what we discussed in the decision-trees module).

Random Forest

Finally I used a Random Forest (RandomForestClassifier from sklearn.ensemble) with 100 trees and a fixed random seed.

A random forest builds many different decision trees on bootstrapped samples of the data, and then averages their predictions. This is an ensemble method, which usually gives better generalization than a single tree.

On the validation set the random forest achieved the best accuracy, about 0.61, clearly beating the baseline, logistic regression, and the single decision tree.

The classification report and confusion matrix showed that the random forest still struggles with the rarest classes (like straight flush and royal flush), but it performs noticeably better across the more common classes.

Final evaluation on the test set

After choosing the random forest as the best model, I retrained each model on the full training set (all 25,010 examples) and evaluated them on the separate test set of 1,000,000 hands:

Baseline (most common class): ~0.501 test accuracy

Logistic Regression: ~0.501 test accuracy

Decision Tree: ~0.478 test accuracy

Random Forest: ~0.608 test accuracy

I also plotted the confusion matrix for the random forest on the test data. The diagonal entries are highest for classes 0 and 1, and much smaller for the rare classes, which matches the strong class imbalance in the dataset. This connects directly to the confusion matrix and precision/recall ideas from Lecture 2.3.

Overall, the modelling approach followed the course structure: start with a simple baseline, then try increasingly powerful models (logistic regression → decision tree → random forest), and compare them using the standard error metrics (accuracy, precision, recall, f1, and confusion matrices) on a held-out validation set and a final test set.

This gives a clean, numeric dataset that is ready to be used with the basic supervised learning methods we covered in the course modules.

Results:

Figure 1:
<img width="713" height="388" alt="image" src="https://github.com/user-attachments/assets/537322e5-0f73-4a5f-bbb7-f66de4a3dc8f" />

Figure 2:
<img width="533" height="455" alt="image" src="https://github.com/user-attachments/assets/0f2d77ac-e56e-4fa0-8d77-c184c5d2894b" />

Figure 3:
<img width="588" height="390" alt="image" src="https://github.com/user-attachments/assets/f6bc7a49-d518-4894-afa1-4eb4656d4fd5" />


Figure 1 shows the distribution of poker hand classes in the training data. Classes 0 (“nothing in hand”) and 1 (“one pair”) make up almost all of the examples. The stronger hands, like straight flush and royal flush (classes 8 and 9), have only a handful of samples. This confirms that the dataset is extremely imbalanced, which already suggests that accuracy alone might be a bit misleading.

Next, I compared four models on the held-out test set of 1,000,000 hands. The test accuracies are summarized in Figure 2:

Baseline (always predict class 0): ≈ 0.50

Logistic Regression: ≈ 0.50

Decision Tree: ≈ 0.48

Random Forest: ≈ 0.61

The baseline and logistic regression perform almost the same, only slightly above 50% accuracy. The decision tree does a little worse overall (≈ 0.48). The random forest clearly performs best, reaching about 0.61 accuracy, which is about 10 percentage points higher than the baseline.

Figure 3 shows the confusion matrix for the random forest on the test set. The diagonal entries for classes 0 and 1 are very large, meaning the model usually predicts those common hands correctly. For the rarer classes (like 5–9), the diagonal values are much smaller and there is more mass in the first few columns. This means that many rare strong hands are still being misclassified as more common weaker hands.

The validation results (on the 20% validation split) follow the same pattern as the test results: random forest has the highest validation accuracy (about 0.61), while logistic regression and the single decision tree stay around 0.5 and 0.49. The classification_report for each model also showed low precision/recall for the rarest classes, especially for models other than the random forest.

Discussion:

The results match what we would expect from both the class imbalance and the types of models used.

First, the high accuracy of the baseline and logistic regression (around 0.50) shows that a model can reach 50% accuracy just by mostly predicting the dominant class (0). This is why it’s important to compare against the baseline. Logistic regression is a linear model, so it struggles to capture the complicated non-linear relationships between card suits/ranks and poker hands, and it mainly learns to separate the common classes.

The decision tree can, in theory, model more complex decision boundaries. However, with this very imbalanced data it tends to overfit the training set and doesn’t generalize as well to unseen data, which explains its lower test accuracy (≈ 0.48). The confusion matrices and classification reports for the tree show that it does not handle the rare classes much better than the simpler models.

The random forest performs best, with test accuracy around 0.61. This makes sense because a random forest is an ensemble of many decision trees trained on different subsets of the data and features. By averaging across trees, it reduces variance and usually generalizes better. In the confusion matrix, we see that the random forest is noticeably better at distinguishing between “nothing”, “one pair”, and “two pair” than the other models. However, it still has very poor recall for the strongest hands (classes 7–9). Most of those rare hands are still predicted as more common classes.

This behavior is consistent with the error-metrics lecture (2.3). Even though overall accuracy improves, the precision and recall for minority classes remain low. The model is biased toward the majority classes, which is a typical problem with class imbalance. To really improve performance on the rare, high-value hands, we would probably need to try techniques such as class-weighted loss functions, oversampling the rare classes, or collecting more examples of those hands.

Overall, the experiment shows that:

-A simple baseline is surprisingly strong when classes are imbalanced.
-Tree-based ensemble methods (random forests) give a clear boost over single trees and linear models.
-Looking at confusion matrices and per-class metrics is crucial; accuracy alone hides the fact that the model almost never recognizes the rarest poker hands correctly.

Conclusion:
For this project I built and compared several machine-learning models to predict the strength of poker hands. A simple baseline model that always predicts “no made hand” (class 0) already reached about 50% test accuracy, just because more than half of all examples belong to that class. This showed me that accuracy by itself can be misleading when the dataset is highly imbalanced, and that every model should first be compared against a basic baseline.

Among the real models, the Random Forest classifier performed the best. On the held-out test set it achieved an accuracy of about 0.61, clearly higher than Logistic Regression (around 0.50) and the single Decision Tree (around 0.48). This suggests that an ensemble of trees is better at capturing the complicated patterns in card ranks and suits than a linear model or a single tree.

However, the confusion matrix made it clear that all models, including the Random Forest, are much better at predicting common hands than rare ones. Classes 0 and 1 (no pair and one pair) are predicted reasonably well, but the model struggles with rare high-value hands such as full houses, four-of-a-kind, and straight flushes, where there are very few training examples. The classification report and confusion matrix were important here, because they showed which classes the model actually understands, not just the overall accuracy.

If I continued this project, I would focus on improving performance on the rare classes. One idea would be to handle the class imbalance more directly, for example by oversampling rare hands, undersampling the majority class, or using class-weighted training so the model pays more attention to rare but important outcomes. Another direction would be to engineer more informative features, such as “number of cards with the same rank,” “is_flush,” “is_straight,” or “highest rank in the hand,” which are closer to how humans think about poker. Finally, I could explore more advanced models (like gradient boosting) and tune hyperparameters, and also report additional metrics such as macro-averaged F1 scores or ROC/AUC for simplified binary versions of the problem. Overall, this project showed that machine learning can learn meaningful patterns from poker data, while also highlighting the limitations that come from imbalanced datasets and rare events.



