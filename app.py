import os
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import nltk
import numpy as np
import pickle
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import time
import datetime
import pytz
from flask_cors import CORS

# Suppress TensorFlow info, warning, and error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download NLTK resources
nltk.download('punkt')

# Create Flask app
app = Flask(__name__)
CORS(app)

# Load the saved model and vectorizer
model = load_model('chatbot_model.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define the corpus and responses
corpus = [
    'Hi there! hiii helo hlo hai hay hello hii',
    'How are you?',
    'What is your name?',
    'Who you are?',
    'what is thikse doing?',
    'Can you help me?',
    'I want your contact number details.',
    'Where is your office located?',
    'How can I reach your HR department?,HR contact,How to contact your HR, HR email id, HR',
    'Who is your founder and CEO?What is your founder name?',
    'When was the organization founded?',
    'How old are you?',
    'What is the weather like today?',
    'Goodbye!',
    'Bye see you',
    'Good morning',
    'Good afternoon',
    'Good evening'
]

responses = [
    'Hello! Happy to welcome you.',
    'I am fine, thank you!',
    'My name is Thiksebot.',
    'Thikse, as a service provider, specializes in transforming dreams into actionable plans and turning those plans into reality.',
    'Thikse, as a service provider, specializes in transforming dreams into actionable plans and turning those plans into reality.',
    'Yes, I can help you. What do you need?',
    'Contact number: +91 4146 358 357.',
    '746/D 2nd floor, Nehruji Road, Villupuram 605602.',
    'For career-related tips, contact our HR at oviya@thikseservices.onmicrosoft.com.',
    'Krishnakanth is the founder and director of the organization.',
    'The organization started in 2024.',
    'I am a Thikse chatbot, so I do not have an age.',
    'The weather today is sunny with a chance of clouds.',
    'Thank you for your visit, Goodbye!',
    'Thank you for your visit, Goodbye! ',
    'Good morning! How can I assist you today?',
    'Good afternoon! How can I assist you today?',
    'Good evening! How can I assist you today?'
]

career_corpus = [
    'What career opportunities',
    'IT, Non-IT, Data Entry, full-stack developer, Front-End, Back-End, AI Developer, ai, hr',
    'Python, JavaScript, HR, Digital Marketing',
    'How can I reach your HR department? HR contact',
    'How can I apply for a job?',
    'Do you offer internships? do you provide any internship',
]

career_responses = [
    'We currently have various career opportunities available in IT\nNon-IT\nData Entry\nFull Stack developer\nFront-End developer\nBack-End developer\nAI developer\nPython\nJavaScript\nHR\nDigital Marketing\nCould you specify your field of interest?',
    'Ok, send your resume to our HR at oviya@thikseservices.onmicrosoft.com.',
    'Ok, send your resume to our HR at oviya@thikseservices.onmicrosoft.com.',
    'For career-related tips, contact our HR at oviya@thikseservices.onmicrosoft.com.',
    'You can apply for a contact our HR at oviya@thikseservices.onmicrosoft.com.',
    'Yes, we offer internships. For more details, please contact our HR at oviya@thikseservices.onmicrosoft.com.',
]

organization_corpus = [
    'what are the service Do you provide ',
    'what is organization name',
    'what is your company name',
    'who your co-founder name cofounder co founder?',
    'who your VicePresident, VP name',
    'VP name Vice President',
    'Tell me more about your organization.',
    'Who are founder ceo in your organization?',
    'What are the values of your organization?',
    'your business partner',
    'what is your motto',
    'Your client details.client name',
]

organization_responses = [
    'We Provide various services: Gen AI Solution\nApp Development\nWeb Development\nCyber Security Services\nDigital Marketing Solutions\nNon IT Services',
    'Our Organization name is "Thikse Software Solution PVT LTD"',
    'Our Organization name is "Thikse Software Solution PVT LTD"',
    "Our Co-Founder is Mr.Shiva\nAs a co-founder of Thikse Software Solutions, I drive our vision forward by fostering innovation and collaboration. I spearhead strategic initiatives, nurture our team's talents, and cultivate a culture of excellence. Together with my co-founders, I am passionate about delivering exceptional solutions to our clients.",
    "Our Vice President is Mrs.Malavika\nAs the Chief Financial Officer, I prioritize my responsibility by placing significant value on the engagement of our people and fostering a motivating environment. I actively support my leadership team in their aspirations and endeavors to achieve targets through effective management of funds and transactions.",
    "Our Vice President is Mrs.Malavika\nAs the Chief Financial Officer, I prioritize my responsibility by placing significant value on the engagement of our people and fostering a motivating environment. I actively support my leadership team in their aspirations and endeavors to achieve targets through effective management of funds and transactions.",
    'Our Mission: At Thikse Software Solutions, we\'re on a mission to empower talent and exceed client expectations. By championing fresh perspectives and securing exciting projects, we\'re building a dynamic community where innovation flourishes and success is inevitable.\n Our Vision: At Thikse Software Solutions, we prioritize quality and timely delivery while fostering our team\'s growth. Our commitment to excellence extends beyond projects—we\'re constantly innovating, expanding, and improving. Join us in creating a future where success knows no bounds.',
    'Our founder and director Mr.Krishnakanth, along with other talented professionals.',
    'Our values include integrity, teamwork, and customer satisfaction.',
    "Business partner is 'Ascent' Our clients include 'IndusInd bank' and 'HDFC Bank'.",
    'Attracting Top Talent',
    "Our clients include 'IndusInd bank' and 'HDFC Bank'",
]

# Define a dictionary for single-word queries
single_word_responses = {
    'who': 'Krishnakanth is the founder and director of the organization.',
    'hello': 'Hello! Happy to welcome you.',
    'hi': 'Hello! Happy to welcome you.',
}

# Define Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message']
        category = request.json['category']

        if category.lower() == 'general':
            response = generate_response(user_input, category, corpus, responses)
        elif category.lower() == 'career':
            response = generate_response(user_input, category, career_corpus, career_responses)
        elif category.lower() == 'organization':
            response = generate_response(user_input, category, organization_corpus, organization_responses)
        else:
            response = "I'm sorry, I don't understand that category."

        return jsonify({'message': response})

    except Exception as e:
        # Log the exception for debugging purposes
        app.logger.error(f"An error occurred: {str(e)}")
        # Return an error response to the client
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500
    
# Function to process user input and generate response
def generate_response(user_input, category, corpus, responses):
    # Define a similarity threshold
    SIMILARITY_THRESHOLD = 0.3

    # Check if the user input is a single word and in the single_word_responses dictionary
    if user_input.lower() in single_word_responses:
        return single_word_responses[user_input.lower()]

    # Combine user input with relevant corpus
    all_corpus = corpus + [user_input]

    # Tokenize and stem the words in the new dataset
    stemmer = PorterStemmer()
    corpus_tokens = [nltk.word_tokenize(sentence.lower()) for sentence in all_corpus]
    corpus_stemmed = [' '.join([stemmer.stem(token) for token in tokens]) for tokens in corpus_tokens]

    # Vectorize the input
    corpus_vectorized = vectorizer.transform(corpus_stemmed)
    user_input_vectorized = corpus_vectorized[-1]

    # Compute cosine similarities
    similarities = cosine_similarity(user_input_vectorized, corpus_vectorized[:-1])

    # Check if the maximum similarity is below the threshold
    if max(similarities[0]) < SIMILARITY_THRESHOLD:
        response = "I'm sorry, I don't understand that."
    else:
        # Predict the response using the model
        prediction = model.predict(user_input_vectorized)
        predicted_label = np.argmax(prediction)

        # Retrieve the response from the label encoder
        response = label_encoder.inverse_transform([predicted_label])[0]

        # Specific response for founder details
        if 'founder' in user_input.lower() or 'ceo' in user_input.lower():
            response = "Krishnakanth is the founder and director of the organization."

    # Adjust greeting based on time of day
    response = adjust_greeting(response, user_input)

    # Set a delay of 1 second for bot response (optional)
    time.sleep(1)

    return response

# Function to adjust greetings based on the response
def adjust_greeting(response, user_input):
    # List of morning, afternoon, and evening greetings
    morning_greetings = ['good morning', 'morning']
    afternoon_greetings = ['good afternoon', 'afternoon']
    evening_greetings = ['good evening', 'evening']

    # Get current time in UTC
    current_time = datetime.datetime.now(pytz.UTC)

    # Convert UTC time to IST (Indian Standard Time)
    current_time = current_time.astimezone(pytz.timezone('Asia/Kolkata'))

    # Extract hour from current time
    current_hour = current_time.hour

    # Adjust greeting based on the current hour
    if any(greeting in user_input.lower() for greeting in morning_greetings):
        response = 'Good morning! How can I assist you today?'
    elif any(greeting in user_input.lower() for greeting in afternoon_greetings):
        response = 'Good afternoon! How can I assist you today?'
    elif any(greeting in user_input.lower() for greeting in evening_greetings):
        response = 'Good evening! How can I assist you today?'

    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
