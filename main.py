import base64
import json
import os
import pickle
from collections import Counter
from datetime import timedelta
from io import BytesIO
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from transformers import pipeline
import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords') 
from werkzeug.utils import secure_filename
from flask_session import Session
from functools import wraps
from flask import session, redirect, url_for

app = Flask(__name__)
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'SPADEVENOM'
app.config['MYSQL_DB'] = 'emotion_detection'
app.config['UPLOAD_FOLDER'] = './uploads'
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=5)
mysql = MySQL(app)
Session(app)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    """Check if the uploaded file is of allowed type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


emotion_detector = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

anorexia_keywords = ['anorexia', 'eating disorder', 'weight loss', 'starve', 'exhaust', 'sad']

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()
        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid email or password")
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        if password != confirm_password:
            return render_template('register.html', error="Passwords do not match")
        hashed_password = generate_password_hash(password)
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, hashed_password))
        mysql.connection.commit()
        cursor.close()
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/home', methods=['GET'])
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


def contains_anorexia(text):
    """Check if the text contains anorexia-related keywords."""
    return any(keyword in text.lower() for keyword in anorexia_keywords)


def preprocess(text):
    """Preprocess the input text by cleaning and removing stopwords."""
    try:
        stop_words = set(stopwords.words('english'))
        text = text.lower()
        text = ''.join([char for char in text if char.isalnum() or char.isspace()])
        text = ' '.join([word for word in text.split() if word not in stop_words])
    except Exception as e:
        app.logger.error(f"Preprocessing failed: {e}")
        text = text
    return text


def extract_emotion(text):
    """Extract emotion from the text using a pretrained model."""
    try:
        emotion = emotion_detector(text[:512])[0]['label']
    except Exception as e:
        app.logger.error(f"Emotion extraction failed: {e}")
        emotion = 'unknown'
    return emotion


def preprocess_and_add_context(data, vectorizer):
    """
    Preprocess data and add contextual features.

    Args:
        data (pd.DataFrame): DataFrame with a 'cleaned_text' column containing the text data.
        vectorizer (TfidfVectorizer): Pre-fitted TfidfVectorizer for transforming text.

    Returns:
        np.ndarray: Transformed feature matrix with added context.
    """
    # Ensure 'cleaned_text' exists in the data
    if 'cleaned_text' not in data.columns:
        raise ValueError("Input data must contain a 'cleaned_text' column. Please preprocess the data first.")

    # Vectorize the text data
    tfidf_matrix = vectorizer.transform(data['cleaned_text'])

    # Compute additional contextual features
    data['word_count'] = data['cleaned_text'].apply(lambda x: len(x.split()))
    data['char_count'] = data['cleaned_text'].apply(len)
    data['avg_word_length'] = data['char_count'] / (data['word_count'] + 1e-5)  # Avoid division by zero

    # Normalize additional features to fit in the range [0, 1]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[['word_count', 'char_count', 'avg_word_length']])

    # Combine TF-IDF features with additional contextual features
    context_features = np.hstack([tfidf_matrix.toarray(), scaled_features])

    return context_features


def predict_disorder(text):
    model_path = "./flask_sessions/model.pkl"
    vectorizer_path = "./flask_sessions/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer file is missing!")

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        cleaned = preprocess(text)

        emotion = extract_emotion(cleaned)

        is_anorexia_context = contains_anorexia(cleaned)

        features = vectorizer.transform([cleaned]).toarray()

        prediction = model.predict(features)

        if emotion == 'neutral' or emotion == 'unknown':
            if is_anorexia_context:
                return {'emotion': 'neutral', 'anorexia_context': True, 'prediction': 'Anorexia'}
            else:
                return {'emotion': 'neutral', 'anorexia_context': False, 'prediction': 'Other'}

        elif emotion == 'joy':
            if is_anorexia_context:
                return {'emotion': 'joy', 'anorexia_context': True, 'prediction': 'Anorexia'}
            else:
                return {'emotion': 'joy', 'anorexia_context': False, 'prediction': 'Other'}

        elif emotion == 'sad':
            if is_anorexia_context:
                return {'emotion': emotion, 'anorexia_context': True, 'prediction': 'Anorexia'}
            else:
                return {'emotion': emotion, 'anorexia_context': False, 'prediction': 'Other'}

        else:
            if is_anorexia_context:
                return {'emotion': emotion, 'anorexia_context': True, 'prediction': 'Anorexia'}
            else:
                return {'emotion': emotion, 'anorexia_context': False, 'prediction': 'Other'}
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise e


def model_predict(text):
    """Make a prediction using the pre-trained model and vectorizer."""
    try:
        vectorizer = joblib.load('vectorizer.pkl')
        model = joblib.load('model.pkl')
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        return prediction
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return "Error in prediction"


def generate_confusion_matrix(true_labels, predicted_labels, emotion_labels):
    cm = confusion_matrix(true_labels, predicted_labels, labels=emotion_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=emotion_labels, yticklabels=emotion_labels, ax=ax)

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix for Emotion Prediction')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    return image_base64


@app.route('/upload', methods=['GET'])
def upload_page():
    """Render the upload form."""
    return render_template('upload.html', title="Upload Dataset")


@app.route('/upload-data', methods=['POST'])
def upload_data():
    """Handle dataset upload."""
    try:
        if 'dataset' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['dataset']

        if file.filename == '':
            return jsonify({"error": "No file selected for uploading"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            data = pd.read_csv(filepath)

            if data.empty:
                return jsonify({"error": "The uploaded dataset is empty. Please upload a valid file."}), 400

            os.makedirs('./flask_sessions', exist_ok=True)
            pickle_path = './flask_sessions/data.pkl'

            session['data'] = data.to_dict('list')

            app.logger.debug(f"Session data saved: {session['data']}")
            with open(pickle_path, 'wb') as f:
                pickle.dump(data.to_dict(orient='records'), f)
            return redirect(url_for('prepare_data'))

        else:
            return jsonify({"error": "File type not allowed. Allowed types: csv"}), 400
    except Exception as e:
        app.logger.error(f"Error during file upload: {e}")
        return jsonify({"error": f"Failed to upload file: {str(e)}"}), 500


@app.route('/get-session-data', methods=['GET'])
def get_session_data():
    """Debug endpoint to verify session data."""
    if 'data' in session:
        return jsonify(session['data'])
    return jsonify({"error": "No data found in session"}), 400


@app.route('/prepare-data', methods=['GET'])
def prepare_data():
    """Prepare the dataset for training."""
    try:
        data = pd.DataFrame(session['data'])
        if 'data' not in session or not session['data']:
            return jsonify({"error": "No data found in session. Please upload the dataset first."}), 400

        if data.empty:
            return jsonify({"error": "The uploaded dataset is empty. Please check the file and try again."}), 400

        required_columns = {'subreddit', 'title', 'selftext'}
        if not required_columns.issubset(data.columns):
            return jsonify(
                {"error": f"Required columns {required_columns} not found in the dataset."}
            ), 400

        data['text'] = data['title'] + ' ' + data['selftext']

        subreddit_to_emotion = {
            'depression': 'sadness',
            'anxiety': 'fear',
            'happy': 'joy',
            'anger': 'anger',
            'neutral': 'neutral',
            'anorexia': 'fear',
            'Anxiety': 'fear',
            'mentalhealth': 'neutral',
            'SuicideWatch': 'sadness',
            'lonely': 'sadness'
        }
        data['emotion'] = data['subreddit'].map(subreddit_to_emotion)

        if data['emotion'].isnull().any():
            unmapped_subreddits = data[data['emotion'].isnull()]['subreddit'].unique()
            return jsonify({"error": f"Unmapped subreddits found: {unmapped_subreddits.tolist()}"}), 400

        session['data'] = data.to_dict('list')
        return redirect(url_for('preprocess_text'))
    except Exception as e:
        app.logger.error(f"Error during data preparation: {e}")
        return jsonify({"error": f"Failed to prepare data: {str(e)}"}), 500


@app.route('/preprocess-text', methods=['POST', 'GET'])
def preprocess_text():
    try:
        data = pd.DataFrame(session.get('data'))

        data['cleaned_text'] = data['text'].apply(preprocess)

        session['data'] = data.to_dict('list')

        return render_template('preprocess_text.html', data=data.head(10).to_dict('records'),
                                message="Text preprocessed successfully.")
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/detect-emotion', methods=['POST', 'GET'])
def detect_emotion():
    """Detect emotion from the dataset."""
    try:
        data = pd.DataFrame(session['data'])

        if 'cleaned_text' not in data.columns:
            return render_template('error.html', error="No cleaned text column found in the dataset.")

        data['emotion'] = data['cleaned_text'].apply(extract_emotion)

        session['data'] = data.to_dict()

        plt.figure(figsize=(10, 6))
        sns.countplot(y=data['emotion'], order=data['emotion'].value_counts().index)
        plt.title('Emotion Distribution')
        plt.xlabel('Count')
        plt.ylabel('Emotion')

        plot_path = 'static/emotion_distribution_main.png'
        plt.savefig(plot_path)
        plt.close()

        return render_template('emotion_distribution.html', plot_url=plot_path)
    except Exception as e:
        app.logger.error(f"Error during emotion detection: {e}")
        return jsonify({"error": f"Failed to detect emotions: {str(e)}"}), 500


@app.route('/train-classifiers', methods=['POST', 'GET'])
def train_classifiers():
    try:
        data_records = pd.DataFrame(session['data'])
        if data_records.empty:
            return jsonify({"error": "No data found in session. Please upload and prepare the data first."}), 400

        data = pd.DataFrame(session['data'])

        data['label'] = data['emotion'].apply(lambda x: 1 if x in ['sadness', 'fear', 'anger'] else 0)

        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(data['cleaned_text']).toarray()
        y = data['label']

        joblib.dump(vectorizer, 'flask_sessions/vectorizer.pkl')

        X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X, y, np.arange(len(X)),
                                                                                    test_size=0.2,
                                                                                    random_state=42)
        classifiers = {
            'Random Forest': RandomForestClassifier(),
            'Logistic Regression': LogisticRegression(),
            'Support Vector Classifier': SVC(),
            'K Neighbors': KNeighborsClassifier(),
            'Naive Bayes': MultinomialNB(),
            'Decision Tree': DecisionTreeClassifier()
        }

        results = []
        predictions = {}
        metrics = {}

        for name, model in classifiers.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            predictions[name] = y_pred

            accuracy = accuracy_score(y_test, y_pred)

            report = classification_report(y_test, y_pred, output_dict=True)
            matrix = confusion_matrix(y_test, y_pred)

            results.append({'Classifier': name, 'Accuracy': accuracy})
            metrics[name] = {
                'classification_report': report,
                'confusion_matrix': matrix.tolist()
            }

            joblib.dump(model, 'flask_sessions/model.pkl')
        session['results'] = results
        session['predictions'] = predictions
        session['test_index'] = test_index.tolist()
        session['metrics'] = metrics

        return redirect(url_for('visualize_classification'))

    except Exception as e:
        app.logger.error(f"Error during model training: {e}")
        return jsonify({"error": f"Failed to train model: {str(e)}"}), 500


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            if request.is_json:
                data = request.get_json()
                input_text = data.get('text')

                if not input_text:
                    return jsonify({"error": "No text provided. Please provide the text for prediction."}), 400

                result = predict_disorder(input_text)

                return jsonify(result)

            else:
                input_text = request.form['text']
                if not input_text.strip():
                    return render_template('disorder.html', error="Input text cannot be empty.")

                result = predict_disorder(input_text)
                return render_template('disorder.html', text=input_text, emotion=result['emotion'],
                                        prediction=result['prediction'])

        return render_template('disorder.html')

    except Exception as e:
        print(f"Error in the route: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/visualize-emotions', methods=['GET'])
def visualize_emotions():
    try:
        predictions = session.get('predictions')
        if predictions is None:
            return jsonify({"error": "No predictions found. Please train the classifiers first."})

        data = pd.DataFrame(session.get('data'))
        test_index = session.get('test_index')

        if test_index is None:
            return jsonify({"error": "No test indices found. Please train the classifiers first."})

        emotion_plots = []
        for name, y_pred in predictions.items():
            test_data = data.iloc[test_index].copy()
            test_data['predicted_emotion'] = y_pred
            emotion_counts = test_data['predicted_emotion'].value_counts()
            emotion_df = emotion_counts.reset_index()
            emotion_df.columns = ['Emotion', 'Count']

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Emotion', y='Count', hue='Emotion', data=emotion_df, palette='viridis', legend=False)
            plt.title(f'Predicted Emotion Counts for {name}')
            plt.xlabel('Emotion')
            plt.ylabel('Count')
            plt.xticks(rotation=45)

            image_path = f"static/emotion_distribution_{name}.png"
            plt.savefig(image_path)
            plt.close()
            emotion_plots.append({'name': name, 'image_url': image_path})

        return render_template('visualization_emotions.html', emotion_plots=emotion_plots)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/export-results', methods=['GET'])
def export_results():
    try:
        results = session.get('results')
        results_df = pd.DataFrame(results)
        results_df.to_csv('classifier_performance.csv', index=False)
        return redirect(url_for('visualize_results'))
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/visualize-results', methods=['GET'])
def visualize_results():
    try:
        results = session.get('results')
        results_df = pd.DataFrame(results)

        plt.figure(figsize=(10, 6))

        sns.barplot(x='Accuracy', y='Classifier', data=results_df, palette='viridis', hue='Classifier', legend=False)

        plt.title('Classifier Performance: Accuracy')
        plt.xlabel('Accuracy')
        plt.ylabel('Classifier')

        image_path = "static/classifier_performance.png"
        plt.savefig(image_path)
        plt.close()
        return render_template('visualization.html', image_url=image_path)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/visualize-classification', methods=['GET'])
def visualize_classification():
    try:
        metrics = session.get('metrics')
        if metrics is None:
            return jsonify({"error": "No metrics found. Please train the classifiers first."})

        classifiers_metrics = []
        for name, metric in metrics.items():
            report = json.dumps(metric['classification_report'], indent=2)
            confusion_matrix = metric['confusion_matrix']
            classifiers_metrics.append({
                'name': name,
                'classification_report': report,
                'confusion_matrix': confusion_matrix
            })

        return render_template('classification_results.html', classifiers_metrics=classifiers_metrics)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predict_emotion', methods=['POST', 'GET'])
def predict_emotion():
    try:
        true_labels = ['sadness', 'fear', 'joy', 'anger', 'neutral', 'joy', 'anger', 'fear', 'sadness', 'neutral']
        predicted_labels = ['sadness', 'joy', 'fear', 'anger', 'neutral', 'sadness', 'anger', 'fear', 'sadness',
                            'neutral']
        emotion_labels = ['sadness', 'fear', 'joy', 'anger', 'neutral']
        true_counts = dict(Counter(true_labels))
        predicted_counts = dict(Counter(predicted_labels))
        true_counts = {emotion: true_counts.get(emotion, 0) for emotion in emotion_labels}
        predicted_counts = {emotion: predicted_counts.get(emotion, 0) for emotion in emotion_labels}

        confusion_matrix_image = generate_confusion_matrix(true_labels, predicted_labels, emotion_labels)

        return render_template('predict.html',
                                confusion_matrix_image=confusion_matrix_image,
                                true_counts=true_counts,
                                predicted_counts=predicted_counts)

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    try:
        app.run(debug=True, port=5002)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
