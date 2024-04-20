from stackapi import StackAPI
import datetime
import time
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
from itertools import combinations
import pymongo

def format_timestamp(timestamp):
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%SZ')

import requests
import time

def get_location_coordinates(location):
    geolocator = Nominatim(user_agent="stackoverflow_scraper")
    if location == "No Location":
        return None, None
    
    # Retry logic with exponential backoff
    max_retries = 3
    retry_delay = 2  # seconds
    timeout = 10  # seconds - Adjust as needed
    for retry_count in range(max_retries):
        try:
            location_data = geolocator.geocode(location, timeout=timeout)
            if location_data:
                return location_data.latitude, location_data.longitude
            else:
                return None, None
        except requests.exceptions.RequestException as e:
            print(f"Error geocoding location '{location}': {e}")
            print(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            retry_delay *= 2  # exponential backoff
    
    # If all retries fail, return None for coordinates
    print(f"Failed to fetch coordinates for location '{location}' after {max_retries} retries.")
    return None, None


def extract_location_from_profile(user_profile):
    if 'location' in user_profile:
        return user_profile['location']
    return None

def generate_tag_combinations(tags):
    if len(tags) == 1:
        return "No tag combinations"
    else:
        return list(combinations(tags, 2))

def save_to_mongodb(data):
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["LLM-db"]
    collection = db["questions"]
    
    # Get the last _id in the collection
    last_id = collection.find_one(sort=[('_id', pymongo.DESCENDING)])
    last_id = last_id['_id'] if last_id else 0
    
    # Enumerate _id and insert data
    for i, item in enumerate(data, start=last_id+1):
        existing_question = collection.find_one({'question_id': item['question_id']})
        if existing_question is None:
            item['_id'] = i
            collection.insert_one(item)


def get_questions_by_tag(tags):
    # API_KEY = '*********'
    SITE = StackAPI('stackoverflow') # key = API_KEY
    question_data_list = []
    
    for tag in tags:
        print('tag: '+tag)
        questions = SITE.fetch('questions', tagged=tag, filter='withbody')
        i = 1
        number_of_questions = str(len(questions['items']))
        for question in questions['items']:
            question_data = {}
            question_data['timestamps'] = format_timestamp(question['creation_date'])
            
            # Use .get() with default value to handle missing keys
            user_id = question['owner'].get('user_id', '')
            if user_id != '':
                display_name = question['owner'].get('display_name', '')
                question_data['owner_id'] = f"https://stackoverflow.com/users/{user_id}/{display_name}"
                
                # Fetch full user profile information
                user_profile = SITE.fetch('users/{ids}', ids=[user_id], filter='default')['items'][0]
                question_data['location'] = extract_location_from_profile(user_profile)
                if question_data['location'] is None:
                    question_data['location'] = "No Location"
                    question_data['latitude'] = "None"
                    question_data['longitude'] = "None"
                else:
                    question_data['latitude'], question_data['longitude'] = get_location_coordinates(question_data['location'])
            else:
                question_data['owner_id'] = 'No Owner ID'
                question_data['location'] = "No Location"
                question_data['latitude'] = "None"
                question_data['longitude'] = "None"

            question_data['votes'] = question['score']
            question_data['views'] = question['view_count']
            question_data['question_id'] = f"question-summary-{question['question_id']}"
            
            print('Question: ' + str(i) + '/' + number_of_questions)
            
            # Extracting text from HTML body
            soup = BeautifulSoup(question.get('body', ''), 'html.parser')
            question_data['question_body'] = soup.get_text()
            
            question_data['question_title'] = question['title']
            question_data['tag'] = ' '.join(question['tags'])  # Convert list of tags to string
            question_data['code_snippet'] = 1 if 'code' in question.get('body', '') else 0
            question_data['comments'] = question.get('comment_count', 0)
            question_data['answers'] = question['answer_count']
            question_data['closed'] = 1 if 'closed_date' in question else 0
            question_data['deleted'] = 1 if 'deleted_date' in question else 0
            question_data['first_answer'] = None
            if question['answer_count'] > 0:
                first_answer = SITE.fetch('questions/{ids}/answers', ids=[question['question_id']], filter='withbody')['items'][0]
                question_data['first_answer'] = format_timestamp(first_answer.get('creation_date'))
            else:
                question_data['first_answer'] = "No answers"
            
            question_data['tag_combinations'] = generate_tag_combinations(question['tags'])
            
            question_data_list.append(question_data)
            
            # Introduce a delay of 1 second between API requests to avoid throttle violation
            time.sleep(0.3)
            i+=1
    
    return question_data_list


tags = [
    'gpt-2', 
    'gpt-3', 
    'gpt-3.5', 
    'gpt-4', 
    'gpt4all', 
    'chatgpt-api', 
    'openai-api', 
    'nlp', 
    'chatbot', 
    'langchain', 
    'huggingface-transformers',
    'large-language-model',
] 
questions_data = get_questions_by_tag(tags)
save_to_mongodb(questions_data)
