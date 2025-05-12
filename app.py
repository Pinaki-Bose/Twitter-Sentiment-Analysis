from flask import Flask, request, render_template
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import csv
import joblib
import plotly.express as px

# Try to download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_csv():
    input_file = 'data/predicted_tweets.csv'
    output_file = 'data/cleaned_predicted_tweets.csv'

    with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        seen_header = False
        writer.writerow(['Tweet', 'Sentiment', 'Sarcasm'])  # Write clean header

        for row in reader:
            # Skip repeated headers
            if row == ['Tweet', 'Sentiment', 'Sarcasm']:
                continue

            if len(row) == 2:  # Add default sarcasm if missing
                row.append('Not Sarcastic')
            if len(row) == 3:
                writer.writerow(row)



def save_prediction(tweet, sentiment, sarcasm):
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/predicted_tweets.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Tweet', 'Sentiment', 'Sarcasm'])
        writer.writerow([tweet, sentiment, sarcasm])

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

sarcasm_model = joblib.load('sarcasm_model.pkl')
sarcasm_vectorizer = joblib.load('sarcasm_vectorizer.pkl')

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
        sarcasm_vectorized = sarcasm_vectorizer.transform([clean_tweet])
        sarcasm_prediction = sarcasm_model.predict(sarcasm_vectorized)
        sarcasm_result = 'Sarcastic' if sarcasm_prediction == 1 else 'Not Sarcastic'
        save_prediction(tweet, result, sarcasm_result)

        return render_template('index.html', tweet=tweet, result=result, sarcasm_result=sarcasm_result)
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    clean_csv()
    predictions = []
    sentiment_count = {'Positive': 0, 'Negative': 0}  # For sentiment distribution
    sarcasm_count = {'Sarcastic': 0, 'Not Sarcastic': 0}  # For sarcasm distribution
    
    # Try reading from the 'predictions.csv' file
    try:
        with open('data/cleaned_predicted_tweets.csv', mode='r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) < 3:
                       continue  # skip bad rows

                tweet = row[0].strip()
                sentiment = row[1].strip()
                sarcasm = row[2].strip()

                predictions.append({'tweet': tweet, 'sentiment': sentiment, 'sarcasm': sarcasm})

                if sentiment.lower() == 'positive':
                         sentiment_count['Positive'] += 1
                else:
                         sentiment_count['Negative'] += 1

                if sarcasm.lower() == 'sarcastic':
                         sarcasm_count['Sarcastic'] += 1
                else:
                         sarcasm_count['Not Sarcastic'] += 1

            print("Sentiment Count:", sentiment_count)
            print("Sarcasm Count:", sarcasm_count)
          
        
    except FileNotFoundError:
        # If the file does not exist yet, just pass
        pass

    # Pie charts for sentiment and sarcasm
    sentiment_pie = px.pie(names=list(sentiment_count.keys()), values=list(sentiment_count.values()), title="Sentiment Distribution")
    sarcasm_pie = px.pie(names=list(sarcasm_count.keys()), values=list(sarcasm_count.values()), title="Sarcasm Distribution")
    
    # Convert plots to HTML components
    sentiment_pie_html = sentiment_pie.to_html(full_html=False)
    sarcasm_pie_html = sarcasm_pie.to_html(full_html=False)

    return render_template('dashboard.html', predictions=predictions, sentiment_pie_html=sentiment_pie_html, sarcasm_pie_html=sarcasm_pie_html)


# Run the app
if __name__ == '__main__':
    app.run( host = '0.0.0.0', port = 5000, debug=False)
