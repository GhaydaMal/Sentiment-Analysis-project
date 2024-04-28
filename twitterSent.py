import pandas as pd
from sklearn.model_selection import train_test_split # split dataset into training and testing sets
from sklearn.feature_extraction.text import TfidfVectorizer #convert text data into numerical feature vectors using TF-IDF
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords # e.g., 'and', 'the', 'is'
from nltk.tokenize import word_tokenize # split text into words or tokens.
from nltk.stem import WordNetLemmatizer # break a word down to its root meaning, to identify similarities.
import string
import nltk

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt') # splits text into sentences and words.
nltk.download('wordnet')#database of English words and their semantic relationships

# Load dataset
data = pd.read_csv("c:\\Users\\ghaid\\Downloads\\Tweets.csv")

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize
    
    # Retain negations
    processed_words = []
    negate = False
    for word in words:
        if word in {'not', 'no', 'nor'}:
            negate = True
        elif word in stop_words:
            if negate:
                processed_words.append('not_' + word)  # Prefix word with "not_" to indicate negation
            else:
                processed_words.append(word)
            negate = False
        else:
            if negate:
                processed_words.append('not_' + word)  # Prefix word with "not_" to indicate negation
            else:
                processed_words.append(word)
    
    return ' '.join(processed_words)

# Adjust column name if necessary
data['clean_text'] = data['text'].apply(preprocess_text)

# Map sentiment labels to numerical values
sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
data['Sentiment'] = data['airline_sentiment'].map(sentiment_mapping)

# Train/Test Split , 20%  testing and 80% trainig 
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['Sentiment'], test_size=0.2, random_state=42) #ensure reproducibility >> same random seed, get the same sequence of random numbers each time

# Define Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()), # converts a collection of raw text data into a matrix
    ('clf', LinearSVC()) #machine learning model used for classification tasks.
])

# Train Model
pipeline.fit(X_train, y_train)

# Evaluate Models' performance
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))

# User Input Loop
while True:
    user_input = input("Enter a sentence: ")
    clean_input = preprocess_text(user_input)

    # Predict
    prediction = pipeline.predict([clean_input])[0]
    if prediction == 1:
        print("The sentiment of the sentence is positive.")
    elif prediction == -1:
        print("The sentiment of the sentence is negative.")
    else:
        print("The sentiment of the sentence is neutral.")
        
    # Ask if the user wants to enter another sentence
    choice = input("Do you want to enter another sentence? (yes/no): ")
    if choice.lower() != 'yes':
        print("Exiting...")
        break
