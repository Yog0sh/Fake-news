.

 Data Analysis
Here I will explain the dataset.

title: this represents the title of the news.
author: this represents the name of the author who has written the news.
text: this column has the news itself.
label: this is a binary column representing if the news is fake (1) or real (0).
The dataset is open-sourced and can be found here.

Libraries
The very basic data science libraries are sklearn, pandas, NumPy e.t.c and some specific libraries such as transformers.

Read dataset from CSV File
df=pd.read_csv('fake-news/train.csv')
df.head()

dataset | detecting fake news NLP
Before proceeding, we need to check whether a null value is present in our dataset or not.

df.isnull().sum()
There is no null value in this dataset. But if you have null values present in your dataset then you can fill it. In the code given below, I will tell you how you can replace the null values.

df = df.fillna(' ')


 Data Preprocessing
In data processing, we will focus on the text column on this data which actually contains the news part. We will modify this text column to extract more information to make the model more predictable. To extract information from the text column, we will use a library, which we know by the name of ‘nltk’.

Here we will use functionalities of the ‘nltk‘ library named Removing Stopwords, Tokenization, and Lemmatization. So we will see these functionalities one by one with these three examples. Hope you will have a better understanding of extracting information from the text column after this.


 Removing Stopwords:-
These are the words that are used in any language used to connect words or used to declare the tense of sentences. This means that if we use these words in any sentence they do not add much meaning to the context of the sentence so even after removing the stopwords we can understand the context.
For more details click on this link.

 Tokenization:-
Tokenization is the process of breaking text into smaller pieces which we know as tokens.
Each word, special character, or number in a sentence can be depicted as a token in NLP.

Tokenization is the process of breaking down a piece of code into smaller units called tokens.


 CONVERTING LABELS:-
The dataset has a Label column whose datatype is Text Category. The Label column in the dataset is classified into two parts, which are denoted as Fake and Real. To train the model, we need to convert the label column to a numerical one.

**********************************************

df.label = df.label.astype(str)
df.label = df.label.str.strip()
dict = { 'REAL' : '1' , 'FAKE' : '0'}
df['label'] = df['label'].map(dict)df.head()
To proceed further, we separate our dataset into features(x_df) and targets(y_df).

x_df = df['total']
y_df = df['label']


VECTORIZATION
Vectorization is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers which is used to find word predictions, word similarities/semantics.

For curiosity, you surely want to check out this article on ‘ Why data are represented as vectors in Data Science Problems’.

To make documents’ corpora more relatable for computers, they must first be converted into some numerical structure. There are few techniques that are used to achieve this such as ‘Bag of Words’.

Here, we are using vectorizer objects provided by Scikit-Learn which are quite reliable right out of the box.

 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count_vectorizer = CountVectorizer()
count_vectorizer.fit_transform(x_df)
freq_term_matrix = count_vectorizer.transform(x_df)
tfidf = TfidfTransformer(norm = "l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)
print(tf_idf_matrix)
Here, with ‘Tfidftransformer’ we are computing word counts using ‘CountVectorizer’ and then computing the IDF values and after that the Tf-IDF scores. With ‘Tfidfvectorizer’ we can do all three steps at once.

The code written above will provide with you a matrix representing your text. It will be a sparse matrix with a large number of elements in a Compressed Sparse Row format.

The most used vectorizers are:

Count Vectorizer: The most straightforward one, it counts the number of times a token shows up in the document and uses this value as its weight.
Hash Vectorizer: This one is designed to be as memory efficient as possible. Instead of storing the tokens as strings, the vectorizer applies the hashing trick to encode them as numerical indexes. The downside of this method is that once vectorized, the features’ names can no longer be retrieved.
TF-IDF Vectorizer: TF-IDF stands for “term frequency-inverse document frequency”, meaning the weight assigned to each token not only depends on its frequency in a document but also how recurrent that term is in the entire corpora. More on that here.

 MODELING
After Vectorization, we split the data into test and train data.

# Splitting the data into test data and train data
x_train, x_test, y_train, y_test = train_test_split(tf_idf_matrix,y_df, random_state=0)
I fit four ML models to the data,

Logistic Regression, Naive-Bayes, Decision Tree, and Passive-Aggressive Classifier.

After that, predicted on the test set from the TfidfVectorizer and calculated the accuracy with accuracy_score() from sklearn. metrics.

. Logistic Regression
#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
Accuracy = logreg.score(x_test, y_test)
print(Accuracy*100)
Accuracy: 91.73%

. Naive-Bayes
#NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(x_train, y_train)
Accuracy = NB.score(x_test, y_test)
print(Accuracy*100)
Accuracy: 82.32 %

Decision Tree
# DECISION TREE

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
Accuracy = clf.score(x_test, y_test)
print(Accuracy*100)
Accuracy: 80.49%

Passive-Aggressive Classifier
Passive Aggressive is considered algorithms that perform online learning (with for example Twitter data). Their characteristic is that they remain passive when dealing with an outcome that has been correctly classified, and become aggressive when a miscalculation takes place, thus constantly self-updating and adjusting.

# PASSIVE-AGGRESSIVE CLASSIFIER

from sklearn.metrics import accuracy_score
from sklearn.linear_model import PassiveAggressiveClassifier
  pac=PassiveAggressiveClassifier(max_iter=50)
 pac.fit(x_train,y_train)
 #Predict on the test set and calculate accuracy
y_pred=pac.predict(x_test)
score=accuracy_score(y_test,y_pred)
 print(f'Accuracy: {round(score*100,2)}%')
