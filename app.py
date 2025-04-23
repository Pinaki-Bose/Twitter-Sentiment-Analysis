from flask import Flask, request, render_template
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Try to download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize stemmer
ps = PorterStemmer()

# Preprocessing function
def preprocess_text(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return ' '.join(review)

# Route for home page
@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    tweet = ''

    if request.method == 'POST':
        tweet = request.form['tweet']
        clean_tweet = preprocess_text(tweet)
        vectorized = vectorizer.transform([clean_tweet])
        prediction = model.predict(vectorized)[0]
        result = 'Positive' if prediction == 1 else 'Negative'

    return render_template('index.html', tweet=tweet, result=result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
