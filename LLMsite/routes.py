from flask import Blueprint, render_template, request, jsonify, redirect, url_for

from flask import Flask

from pymongo import MongoClient

from itertools import islice, combinations

from collections import Counter

import re

from datetime import datetime, timezone, date

import numpy as np

from operator import itemgetter

from statistics import median

import numpy as np

import pandas as pd

import joblib

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

from bertopic import BERTopic

from umap import UMAP

from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from scipy.stats import mannwhitneyu

app = Flask(__name__)

documents = pd.read_csv("static/models/LLM-db.questions.csv")
model = BERTopic.load("static/models/Bertopic_model_reduced_30_topics")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
model_class = joblib.load('static/models/random_forest_model.joblib')

####Predict the topic and embedding of a list of texts (new_docs)
####In the tool use this function to predict the topic of one text
####Remember that topic -1 means that the text was not assigned to a topic and the rest responses indicate a topic assignement
def predict_topic_new_docs(new_docs):
    
    new_embs = sentence_model.encode(new_docs, show_progress_bar=True)
    new_docs_topics=model.transform(new_docs,new_embs)
    return ({"topic":new_docs_topics})

####Predict the texts from a list of texts (new_docs) relevant to ChatGPT
####Returns the class of the texts, 1 for ChatGPT and 0 for non ChatGPT, and their probability being related to ChatGPT (second column) and to non ChatGPT (first column)
####In the tool use this function to predict the class and probability of one text
def predict_class_new_docs(new_docs):

    new_embs = sentence_model.encode(new_docs, show_progress_bar=True)
    new_docs_class = model_class.predict(new_embs)
    new_docs_class_prob = model_class.predict_proba(new_embs)
    result_dict = {}
    for i, doc in enumerate(new_docs):
        result_dict[doc] = [new_docs_class[i], new_docs_class_prob[i][0], new_docs_class_prob[i][1]]
    return result_dict

#####Returns two pandas dataframes relevant to topic popularity (tp_pop_mat) and difficulty (tp_dif_mat)
#####Two dates are used to calculate the popularity and difficulty of the topics within that time span
#####date_format = "%Y-%m-%d %H:%M:%S%z"
def topic_popularity_difficulty(earliest_date,latest_date):

    topic_assignemnts=pd.DataFrame(model.topics_)
    topic_assignemnts=topic_assignemnts.loc[documents['timestamps'] >= earliest_date,:]
    topic_assignemnts=topic_assignemnts.loc[documents['timestamps'] <= latest_date,:]


    df=documents.loc[documents['timestamps'] >= earliest_date  ,:]
    df=df.loc[documents['timestamps'] <= latest_date ,:]


    tp_pop_mat=pd.DataFrame(0,columns=['Avg_views','Avg_score','Avg_comments'],index=range(len(model.topic_sizes_)))
    tp_pop_df_cols=['views','votes','comments']


    tp_dif_mat= pd.DataFrame(0,columns=['Avg_answers','Perc_with_acc','Avg_hrs_to_first_answer',"PD"],index=range(len(model.topic_sizes_)))


    date_string = "2024-04-01 13:56:21Z"
    date_format = "%Y-%m-%d %H:%M:%S%z"

    for i in range(len(tp_pop_mat)):
        df_topic=df.loc[topic_assignemnts[0]==(i-1),]

        if (len(df_topic) > 0):
            for j in range(len(tp_pop_mat.columns)):
                tp_pop_mat.loc[tp_pop_mat.index[i], tp_pop_mat.columns[j]] = sum(df_topic[tp_pop_df_cols[j]]) / len(df_topic)

            tp_dif_mat.loc[tp_pop_mat.index[i], 'Avg_answers'] = sum(df_topic['answers']) / len(df_topic)
            tp_dif_mat.loc[tp_pop_mat.index[i], 'PD'] = (tp_dif_mat.loc[tp_dif_mat.index[i],'Avg_answers']/tp_pop_mat.loc[tp_pop_mat.index[i],'Avg_views'] )*100

            df_topic_first_ans=df_topic.loc[df_topic['answers']>0,["timestamps","first_answer"]]#["timestamp","first_answer"]
            if (len(df_topic_first_ans)>0):
                date_list=[]
                for j in range(len(df_topic_first_ans)):
                    if (df_topic_first_ans['first_answer'][df_topic_first_ans.index[j]]!="No answers"):
                        date_first = datetime.strptime(df_topic_first_ans['timestamps'][df_topic_first_ans.index[j]],date_format)
                        date_second = datetime.strptime(df_topic_first_ans['first_answer'][df_topic_first_ans.index[j]],date_format)
                        difference = date_second - date_first
                        date_list.append(difference.total_seconds())
                if(len(date_list)!=0):
                    tp_dif_mat['Avg_hrs_to_first_answer'][tp_dif_mat.index[i]] = sum(date_list) / (3600 * len(date_list))
    return(tp_pop_mat,tp_dif_mat)

#Returns a dictionary containing information relevant to the growth of the topics according to their overall prevalence in two different dates
#topic_length_early = Number of questions belonging to each topic until the earliest date
#topic_length_late = Number of questions belonging to each topic until the latest date
#share_early = The percentage of questions belonging to each topic until the earliest date
#share_late = The percentage of questions belonging to each topic until the latest date
#self_growth = The percentage that a topic has grown within the two dates
#all_growth = The percentage of the change in the overall prevalence of each topic compared to the rest topics
#####date_format = "%Y-%m-%d %H:%M:%S%z"
def growth_topic(earliest_date,latest_date):

    topic_assignemnts = pd.DataFrame(model.topics_)
    topic_assignemnts_early=topic_assignemnts.loc[documents['timestamps'] <= earliest_date, :]
    topic_assignemnts_late=topic_assignemnts.loc[documents['timestamps'] <= latest_date, :]


    share_early=[]
    share_late=[]
    self_growth=[]
    all_perc=[]
    len_early_list=[]
    len_late_list=[]

    for i in range(model.nr_topics):
        len_early=len(topic_assignemnts_early.loc[topic_assignemnts_early[0]==(i-1),])
        len_early_list.append(len_early)

        len_late=len(topic_assignemnts_late.loc[topic_assignemnts_late[0]==(i-1),])
        len_late_list.append(len_late)
        if(len_early != 0  and len_late !=0):
            share_early.append(len_early/len(topic_assignemnts_early))
            share_late.append(len_late/len(topic_assignemnts_late))

            self_growth.append((len_late-len_early)/len_early)

            perc_early=len_early/len(topic_assignemnts_early)
            perc_late=len_late/len(topic_assignemnts_late)

            all_perc.append((perc_late-perc_early))#/perc_early)
        else:
            share_early.append(0)
            share_late.append(0)
            self_growth.append(0)
            perc_early = 0
            perc_late = 0
            all_perc.append(0)  # /perc_early)

    return({"topic_length_early":len_early_list,"topic_length_late":len_late_list,"share_early":share_early,"share_late":share_late,"self_grown":self_growth,"all_growth":all_perc})

####Compare two topics with , topic_no_1 and topic_no_2 using the MannWhitney U test with respect to a metric and two dates
####Topic_no_1 and topic_no_2 must be numerical
####metric is a list of numerical values
####earliest and latest date are defined as previously
####Returns a dictionary with the pvalues (P-value) , U statistic (Statistic) and the alternative hypothesis of three MannWhitney U tests
#####date_format = "%Y-%m-%d %H:%M:%S%z"
def pairwise_topic_comparisons(topic_no_1,topic_no_2,metric,earliest_date,latest_date):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html
    topic_assignemnts = pd.DataFrame(model.topics_)

    topic_assignemnts_1=[]
    topic_assignemnts_2=[]

    for i in range(len(documents)):
        if documents["timestamps"][documents.index[i]]>=earliest_date:
            if documents["timestamps"][documents.index[i]] <= latest_date:
                if topic_assignemnts[0][i]==topic_no_1:
                    topic_assignemnts_1.append(metric[i])
                elif topic_assignemnts[0][i]==topic_no_2:
                    topic_assignemnts_2.append(metric[i])

    res_G=mannwhitneyu(topic_assignemnts_1,topic_assignemnts_2,alternative="greater")
    res_L=mannwhitneyu(topic_assignemnts_1,topic_assignemnts_2,alternative="less")
    res_two=mannwhitneyu(topic_assignemnts_1,topic_assignemnts_2,alternative="two-sided")

    test_list=[f"Topic {topic_no_1} greater than {topic_no_2}",f"Topic {topic_no_2} greater than {topic_no_1}"]
    pvalue_list=[res_G.pvalue,res_L.pvalue]
    statistic_list=[res_G.statistic,res_L.statistic]

    return({"Alternative Hypothesis":test_list,"P-value":pvalue_list,"Statistic":statistic_list})

@app.route('/', methods=['GET', 'POST'])
def index():
    client = MongoClient('localhost',27017)
    db = client['LLM-db']
    covidCollection = db.questions
    questions = covidCollection.find()  # load questions from collection
    all_questions = list(questions)
    closed_questions = list([q for q in all_questions if q['closed'] == 1])
    deleted_questions = list([q for q in all_questions if q['deleted'] == 1])
    questions = list([q for q in all_questions if q['closed'] == 0 and q['deleted'] == 0])

    techCollection = db.technologies_list
    technologies = techCollection.find()
    users = len(covidCollection.distinct('owner_id'))
    question_number = len(covidCollection.distinct('_id'))
    dates = []
    date_from = '2020-01-01'
    today = date.today()
    date_to = str(today)
    dates_and_values = {}
    tags = []
    tags_and_values = {}
    list_of_tags_and_values = []
    latitudes = []
    longitudes = []
    coordinates = []
    comments = []
    answers = []
    votes = []
    code_snippets = []
    ids_and_votes = {}
    ids_and_answers = {}
    ids_and_comments = {}
    ids_and_views = {}
    ids_and_response_times = {}
    usernames = []
    locations = []
    location_name = []
    location_question = []
    languages = {}
    web_frameworks = {}
    big_data_ml = {}
    databases = {}
    platforms = {}
    collaboration_tools = {}
    dev_tools = {}
    elapsed_time_data_list = []
    languages_elapsed_time_data_list = []
    web_frameworks_elapsed_time_data_list = []
    big_data_ml_elapsed_time_data_list = []
    databases_elapsed_time_data_list = []
    platforms_elapsed_time_data_list = []
    collaboration_tools_elapsed_time_data_list = []
    dev_tools_elapsed_time_data_list = []
    comments_distribution = {}
    votes_distribution = {}
    answers_distribution = {}
    views_distribution = {}
    tag_combo_frequencies = {}
    answered_questions = 0
    unanswered_questions = 0

    fields_and_techs = {}
    for technology in technologies:
        fields_and_techs.update({technology['field'].lower(): technology['technology']})

    question_count = 0
    for question in questions:
        question_count += 1
        dates.append(question['timestamps'][:10])
        record_tags = question['tag'].split()
        dif_tags = list(set(record_tags))

        if question['owner_id'] != 'No Owner ID':
            usernames.append(question['owner_id'])
               
        comments.append(question['comments'])
        if question['comments'] in comments_distribution.keys():
            comments_distribution[question['comments']]+=1
        else:
            comments_distribution[int(question['comments'])]= 1
            
        answers.append(int(question['answers']))
        if question['answers'] in answers_distribution.keys():
            answers_distribution[int(question['answers'])] += 1
        else:
            answers_distribution[int(question['answers'])] = 1
        if int(question['answers']) > 0:
            answered_questions += 1
        else:
            unanswered_questions += 1

        integer_votes = int(question['votes'])
        votes.append(integer_votes)
        if integer_votes in votes_distribution.keys():
            votes_distribution[integer_votes] += 1
        else:
            votes_distribution[integer_votes] = 1
            
        views_text = str((question['views'])).replace(",", "")
        views = re.findall('[0-9]+', views_text)
        
        try:
            if question['tag_combinations'] != 'No tag combinations':
                if len(question['tag_combinations']) < 1000:
                    for tag_combo in question['tag_combinations']:
                        tag_combo_string = ' '.join(tag_combo)
                        if not tag_combo_string in tag_combo_frequencies:
                            tag_combo_frequencies[tag_combo_string] = 1
                        else:
                            tag_combo_frequencies[tag_combo_string] += 1    
        except:
            pass
        
        if len(views) != 0:
            views_integer = int(views[0])
            if views_integer in views_distribution.keys():
                views_distribution[views_integer] += 1
            else:
                views_distribution.update({views_integer : 1})
        
        q_id = question['question_id']
        q_link = "https://stackoverflow.com/questions/" + str(re.sub("[^0-9]", "", q_id))
        ids_and_votes.update({q_link: [int(question['votes']), question['question_title']]})
        ids_and_answers.update({q_link: [int(question['answers']), question['question_title']]})
        ids_and_comments.update({q_link: [int(question['comments']), question['question_title']]})
        ids_and_views.update({q_link: [views_integer, question['question_title']]})
        #########################
        
        question_time = datetime.fromisoformat(question['timestamps'][:-1]).replace(tzinfo=timezone.utc)
        if question_time:           
            if question['first_answer'] != 'No answers':
                first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                event = 1   
            else:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                event = 0    
            if hour_diff!=0.0:
                elapsed_time_data_list.append([hour_diff,event])
                
        ids_and_response_times.update({q_link: [hour_diff, question['question_title']]})

        merged_dict = {}
        for key in ids_and_response_times.keys():
            merged_dict[key] = [ids_and_response_times[key][0], ids_and_response_times[key][1], ids_and_answers[key][0]]
        #########################
        for q_tag in dif_tags:
            if fields_and_techs.get(q_tag) == 'Languages':
                languages.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                        event = 1   
                    else:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                        current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        languages_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Web Frameworks':
                web_frameworks.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                        event = 1   
                    else:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                        current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        web_frameworks_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Big Data - ML':
                big_data_ml.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                        event = 1   
                    else:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                        current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        big_data_ml_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Databases':
                databases.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                        event = 1   
                    else:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                        current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        databases_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Platforms':
                platforms.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                        event = 1   
                    else:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                        current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        platforms_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Collaboration Tools':
                collaboration_tools.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                        event = 1   
                    else:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                        current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        collaboration_tools_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Developer Tools':
                dev_tools.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                        event = 1   
                    else:
                        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"Z"
                        current_time_formatted = datetime.fromisoformat(current_time[:-1]).replace(tzinfo=timezone.utc)
                        hour_diff = round((current_time_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        dev_tools_elapsed_time_data_list.append([hour_diff,event])
        #########################
        code_snippets.append(int(question['code_snippet']))
        if question['latitude'] != 'None':
            latLng = [question['latitude'], question['longitude']]
            coordinates.append(latLng)
            latitudes.append(question['latitude'])
            longitudes.append(question['longitude'])
            locations.append(question['location'])
        for i in range(len(record_tags)):
            tags.append(record_tags[i])
            
    answeredData = [answered_questions, unanswered_questions]
            
    sorted_comments_distribution = dict(sorted(comments_distribution.items(), key=lambda x:x[0]))
    sorted_comments_distribution_labels = list(sorted_comments_distribution.keys())
    sorted_comments_distribution_values = list(sorted_comments_distribution.values())

    sorted_answers_distribution = dict(sorted(answers_distribution.items(), key=lambda x:x[0]))
    sorted_answers_distribution_labels = list(sorted_answers_distribution.keys())
    sorted_answers_distribution_values = list(sorted_answers_distribution.values())

    sorted_votes_distribution = dict(sorted(votes_distribution.items(), key=lambda x:x[0]))
    sorted_votes_distribution_labels = list(sorted_votes_distribution.keys())
    sorted_votes_distribution_values = list(sorted_votes_distribution.values())
    
    sorted_views_distribution = dict(sorted(views_distribution.items(), key=lambda x:x[0]))
    sorted_views_distribution_labels = list(sorted_views_distribution.keys())
    sorted_views_distribution_values = list(sorted_views_distribution.values())
    
    
    sorted_elapsed_time_data_list = sorted(elapsed_time_data_list, key=itemgetter(0))
    languages_sorted_elapsed_time_data_list = sorted(languages_elapsed_time_data_list, key=itemgetter(0))
    web_frameworks_sorted_elapsed_time_data_list = sorted(web_frameworks_elapsed_time_data_list, key=itemgetter(0))
    big_data_ml_sorted_elapsed_time_data_list = sorted(big_data_ml_elapsed_time_data_list, key=itemgetter(0))
    databases_sorted_elapsed_time_data_list = sorted(databases_elapsed_time_data_list, key=itemgetter(0))
    platforms_sorted_elapsed_time_data_list = sorted(platforms_elapsed_time_data_list, key=itemgetter(0))
    collaboration_tools_sorted_elapsed_time_data_list = sorted(collaboration_tools_elapsed_time_data_list, key=itemgetter(0))
    dev_tools_sorted_elapsed_time_data_list = sorted(dev_tools_elapsed_time_data_list, key=itemgetter(0))
    
    elapsed_times_all = []
    elapsed_times_languages = []
    elapsed_times_web_frameworks = []
    elapsed_times_big_data_ml = []
    elapsed_times_databases = []
    elapsed_times_platforms = []
    elapsed_times_colab_tools = []
    elapsed_times_dev_tools = []
    
    for time_list in sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_all.append(time_list[0])
    
    for time_list in languages_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_languages.append(time_list[0])
    
    for time_list in web_frameworks_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_web_frameworks.append(time_list[0])
    
    for time_list in big_data_ml_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_big_data_ml.append(time_list[0])        
    
    for time_list in databases_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_databases.append(time_list[0])
    
    for time_list in platforms_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_platforms.append(time_list[0])
    
    for time_list in collaboration_tools_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_colab_tools.append(time_list[0])
            
    for time_list in dev_tools_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_dev_tools.append(time_list[0])
    
    median_times_dict = {
        "All": median(elapsed_times_all) if elapsed_times_all else 0,
        "Languages": median(elapsed_times_languages) if elapsed_times_languages else 0,
        "Web Frameworks": median(elapsed_times_web_frameworks) if elapsed_times_web_frameworks else 0,
        "Big Data - ML": median(elapsed_times_big_data_ml) if elapsed_times_big_data_ml else 0,
        "Databases": median(elapsed_times_databases) if elapsed_times_databases else 0,
        "Platforms": median(elapsed_times_platforms) if elapsed_times_platforms else 0,
        "Collaboration Tools": median(elapsed_times_colab_tools) if elapsed_times_colab_tools else 0,
        "Developer Tools": median(elapsed_times_dev_tools) if elapsed_times_dev_tools else 0,
    }
    
    times_data_list = []
    number_of_distinct_times_data_list = []
    censored_data_list = []
    times_left_list = []
    
    item_counter = 1
    times_data_list.append(0)
    number_of_distinct_times_data_list.append(0)
    censored_data_list.append(0)
    times_left_list.append(len(sorted_elapsed_time_data_list))
    
    for item in sorted_elapsed_time_data_list:
        if item[0] not in times_data_list :
            if item[1] == 1:
                times_data_list.append(item[0])
                number_of_distinct_times_data_list.append(1)
                censored_data_list.append(0) 
                times_left_list.append(times_left_list[item_counter - 1] - (number_of_distinct_times_data_list[item_counter - 1] + censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                censored_data_list[item_counter-1]+=1
        
    survival_time_curve_values = []
    for i in range(len(times_data_list)):
        if i == 0:
            survival_time_curve_values.append((times_left_list[i] - number_of_distinct_times_data_list[i]) /  times_left_list[i]) if times_left_list[i] != 0 else 0
        else:
            survival_time_curve_values.append(((times_left_list[i] - number_of_distinct_times_data_list[i]) /  times_left_list[i])* survival_time_curve_values[i-1])
    
    
    
    languages_times_data_list = []
    languages_number_of_distinct_times_data_list = []
    languages_censored_data_list = []
    languages_times_left_list = []

    item_counter = 1
    languages_times_data_list.append(0)
    languages_number_of_distinct_times_data_list.append(0)
    languages_censored_data_list.append(0)
    languages_times_left_list.append(len(languages_sorted_elapsed_time_data_list))

    for item in languages_sorted_elapsed_time_data_list:
        if item[0] not in languages_times_data_list :
            if item[1] == 1:
                languages_times_data_list.append(item[0])
                languages_number_of_distinct_times_data_list.append(1)
                languages_censored_data_list.append(0) 
                languages_times_left_list.append(languages_times_left_list[item_counter - 1] - (languages_number_of_distinct_times_data_list[item_counter - 1] + languages_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                languages_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                languages_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                languages_censored_data_list[item_counter-1]+=1

    languages_survival_time_curve_values = []
    for i in range(len(languages_times_data_list)):
        if i == 0:
            languages_survival_time_curve_values.append((languages_times_left_list[i] - languages_number_of_distinct_times_data_list[i]) /  languages_times_left_list[i]) if languages_times_left_list[i] != 0 else 0
        else:
            languages_survival_time_curve_values.append(((languages_times_left_list[i] - languages_number_of_distinct_times_data_list[i]) /  languages_times_left_list[i]) * languages_survival_time_curve_values[i-1])
        
    
    
    web_frameworks_times_data_list = []
    web_frameworks_number_of_distinct_times_data_list = []
    web_frameworks_censored_data_list = []
    web_frameworks_times_left_list = []

    item_counter = 1
    web_frameworks_times_data_list.append(0)
    web_frameworks_number_of_distinct_times_data_list.append(0)
    web_frameworks_censored_data_list.append(0)
    web_frameworks_times_left_list.append(len(web_frameworks_sorted_elapsed_time_data_list))

    for item in web_frameworks_sorted_elapsed_time_data_list:
        if item[0] not in web_frameworks_times_data_list :
            if item[1] == 1:
                web_frameworks_times_data_list.append(item[0])
                web_frameworks_number_of_distinct_times_data_list.append(1)
                web_frameworks_censored_data_list.append(0) 
                web_frameworks_times_left_list.append(web_frameworks_times_left_list[item_counter - 1] - (web_frameworks_number_of_distinct_times_data_list[item_counter - 1] + web_frameworks_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                web_frameworks_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                web_frameworks_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                web_frameworks_censored_data_list[item_counter-1]+=1
                
    web_frameworks_survival_time_curve_values = []
    for i in range(len(web_frameworks_times_data_list)):
        if i == 0:
            web_frameworks_survival_time_curve_values.append((web_frameworks_times_left_list[i] - web_frameworks_number_of_distinct_times_data_list[i]) /  web_frameworks_times_left_list[i]) if web_frameworks_times_left_list[i] != 0 else 0
        else:
            web_frameworks_survival_time_curve_values.append(((web_frameworks_times_left_list[i] - web_frameworks_number_of_distinct_times_data_list[i]) /  web_frameworks_times_left_list[i])* web_frameworks_survival_time_curve_values[i-1])
        
        

    big_data_ml_times_data_list = []
    big_data_ml_number_of_distinct_times_data_list = []
    big_data_ml_censored_data_list = []
    big_data_ml_times_left_list = []

    item_counter = 1
    big_data_ml_times_data_list.append(0)
    big_data_ml_number_of_distinct_times_data_list.append(0)
    big_data_ml_censored_data_list.append(0)
    big_data_ml_times_left_list.append(len(big_data_ml_sorted_elapsed_time_data_list))

    for item in big_data_ml_sorted_elapsed_time_data_list:
        if item[0] not in big_data_ml_times_data_list :
            if item[1] == 1:
                big_data_ml_times_data_list.append(item[0])
                big_data_ml_number_of_distinct_times_data_list.append(1)
                big_data_ml_censored_data_list.append(0) 
                big_data_ml_times_left_list.append(big_data_ml_times_left_list[item_counter - 1] - (big_data_ml_number_of_distinct_times_data_list[item_counter - 1] + big_data_ml_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                big_data_ml_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                big_data_ml_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                big_data_ml_censored_data_list[item_counter-1]+=1
                
    big_data_ml_survival_time_curve_values = []
    for i in range(len(big_data_ml_times_data_list)):
        if i == 0:
            big_data_ml_survival_time_curve_values.append((big_data_ml_times_left_list[i] - big_data_ml_number_of_distinct_times_data_list[i]) /  big_data_ml_times_left_list[i]) if big_data_ml_times_left_list[i] != 0 else 0
        else:
            big_data_ml_survival_time_curve_values.append(((big_data_ml_times_left_list[i] - big_data_ml_number_of_distinct_times_data_list[i]) /  big_data_ml_times_left_list[i])* big_data_ml_survival_time_curve_values[i-1])
            
    databases_times_data_list = []
    databases_number_of_distinct_times_data_list = []
    databases_censored_data_list = []
    databases_times_left_list = []

    item_counter = 1
    databases_times_data_list.append(0)
    databases_number_of_distinct_times_data_list.append(0)
    databases_censored_data_list.append(0)
    databases_times_left_list.append(len(databases_sorted_elapsed_time_data_list))

    for item in databases_sorted_elapsed_time_data_list:
        if item[0] not in databases_times_data_list :
            if item[1] == 1:
                databases_times_data_list.append(item[0])
                databases_number_of_distinct_times_data_list.append(1)
                databases_censored_data_list.append(0) 
                databases_times_left_list.append(databases_times_left_list[item_counter - 1] - (databases_number_of_distinct_times_data_list[item_counter - 1] + databases_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                databases_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                databases_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                databases_censored_data_list[item_counter-1]+=1
                
    databases_survival_time_curve_values = []
    for i in range(len(databases_times_data_list)):
        if i == 0:
            databases_survival_time_curve_values.append((databases_times_left_list[i] - databases_number_of_distinct_times_data_list[i]) /  databases_times_left_list[i]) if databases_times_left_list[i] != 0 else 0
        else:
            databases_survival_time_curve_values.append(((databases_times_left_list[i] - databases_number_of_distinct_times_data_list[i]) /  databases_times_left_list[i])* databases_survival_time_curve_values[i-1])
            
    
    platforms_times_data_list = []
    platforms_number_of_distinct_times_data_list = []
    platforms_censored_data_list = []
    platforms_times_left_list = []

    item_counter = 1
    platforms_times_data_list.append(0)
    platforms_number_of_distinct_times_data_list.append(0)
    platforms_censored_data_list.append(0)
    platforms_times_left_list.append(len(platforms_sorted_elapsed_time_data_list))

    for item in platforms_sorted_elapsed_time_data_list:
        if item[0] not in platforms_times_data_list :
            if item[1] == 1:
                platforms_times_data_list.append(item[0])
                platforms_number_of_distinct_times_data_list.append(1)
                platforms_censored_data_list.append(0) 
                platforms_times_left_list.append(platforms_times_left_list[item_counter - 1] - (platforms_number_of_distinct_times_data_list[item_counter - 1] + platforms_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                platforms_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                platforms_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                platforms_censored_data_list[item_counter-1]+=1
                
    platforms_survival_time_curve_values = []
    for i in range(len(platforms_times_data_list)):
        if i == 0:
            platforms_survival_time_curve_values.append((platforms_times_left_list[i] - platforms_number_of_distinct_times_data_list[i]) /  platforms_times_left_list[i]) if platforms_times_left_list[i] != 0 else 0
        else:
            platforms_survival_time_curve_values.append(((platforms_times_left_list[i] - platforms_number_of_distinct_times_data_list[i]) /  platforms_times_left_list[i])* platforms_survival_time_curve_values[i-1])
    
    collaboration_tools_times_data_list = []
    collaboration_tools_number_of_distinct_times_data_list = []
    collaboration_tools_censored_data_list = []
    collaboration_tools_times_left_list = []

    item_counter = 1
    collaboration_tools_times_data_list.append(0)
    collaboration_tools_number_of_distinct_times_data_list.append(0)
    collaboration_tools_censored_data_list.append(0)
    collaboration_tools_times_left_list.append(len(collaboration_tools_sorted_elapsed_time_data_list))

    for item in collaboration_tools_sorted_elapsed_time_data_list:
        if item[0] not in collaboration_tools_times_data_list :
            if item[1] == 1:
                collaboration_tools_times_data_list.append(item[0])
                collaboration_tools_number_of_distinct_times_data_list.append(1)
                collaboration_tools_censored_data_list.append(0) 
                collaboration_tools_times_left_list.append(collaboration_tools_times_left_list[item_counter - 1] - (collaboration_tools_number_of_distinct_times_data_list[item_counter - 1] + collaboration_tools_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                collaboration_tools_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                collaboration_tools_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                collaboration_tools_censored_data_list[item_counter-1]+=1
                
    collaboration_tools_survival_time_curve_values = []
    for i in range(len(collaboration_tools_times_data_list)):
        if i == 0:
            collaboration_tools_survival_time_curve_values.append((collaboration_tools_times_left_list[i] - collaboration_tools_number_of_distinct_times_data_list[i]) /  collaboration_tools_times_left_list[i]) if collaboration_tools_times_left_list[i] != 0 else 0
        else:
            collaboration_tools_survival_time_curve_values.append(((collaboration_tools_times_left_list[i] - collaboration_tools_number_of_distinct_times_data_list[i]) /  collaboration_tools_times_left_list[i])* collaboration_tools_survival_time_curve_values[i-1])
        
        
    dev_tools_times_data_list = []
    dev_tools_number_of_distinct_times_data_list = []
    dev_tools_censored_data_list = []
    dev_tools_times_left_list = []

    item_counter = 1
    dev_tools_times_data_list.append(0)
    dev_tools_number_of_distinct_times_data_list.append(0)
    dev_tools_censored_data_list.append(0)
    dev_tools_times_left_list.append(len(dev_tools_sorted_elapsed_time_data_list))

    for item in dev_tools_sorted_elapsed_time_data_list:
        if item[0] not in dev_tools_times_data_list :
            if item[1] == 1:
                dev_tools_times_data_list.append(item[0])
                dev_tools_number_of_distinct_times_data_list.append(1)
                dev_tools_censored_data_list.append(0) 
                dev_tools_times_left_list.append(dev_tools_times_left_list[item_counter - 1] - (dev_tools_number_of_distinct_times_data_list[item_counter - 1] + dev_tools_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                dev_tools_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                dev_tools_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                dev_tools_censored_data_list[item_counter-1]+=1
                
    dev_tools_survival_time_curve_values = []
    for i in range(len(dev_tools_times_data_list)):
        if i == 0:
            dev_tools_survival_time_curve_values.append((dev_tools_times_left_list[i] - dev_tools_number_of_distinct_times_data_list[i]) /  dev_tools_times_left_list[i]) if dev_tools_times_left_list[i] != 0 else 0
        else:
            dev_tools_survival_time_curve_values.append(((dev_tools_times_left_list[i] - dev_tools_number_of_distinct_times_data_list[i]) /  dev_tools_times_left_list[i])* dev_tools_survival_time_curve_values[i-1])    
        
    
             
    distinct_locations = Counter(locations)
    for key, value in distinct_locations.items():
        location_name.append(key)
        location_question.append(value)


    ############################
    sorted_language_votes = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_languages_votes = dict(islice(sorted_language_votes.items(), 10))

    sorted_language_answers = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_languages_answers = dict(islice(sorted_language_answers.items(), 10))

    sorted_language_comments = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_languages_comments = dict(islice(sorted_language_comments.items(), 10))

    sorted_language_views = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_languages_views = dict(islice(sorted_language_views.items(), 10))

    answered_languages = {k: v for k, v in languages.items() if v[1] > 0}

    sorted_language_response_time = dict(sorted(answered_languages.items(), key=lambda item: item[1][4]))
    top_10_languages_response_time = dict(islice(sorted_language_response_time.items(), 10))

    sorted_language_response_time_reverse = dict(sorted(answered_languages.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_languages_response_time_reverse = dict(islice(sorted_language_response_time_reverse.items(), 10))

    sorted_web_frameworks_votes = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_web_frameworks_votes = dict(islice(sorted_web_frameworks_votes.items(), 10))

    sorted_web_frameworks_answers = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_web_frameworks_answers = dict(islice(sorted_web_frameworks_answers.items(), 10))

    sorted_web_frameworks_comments = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_web_frameworks_comments = dict(islice(sorted_web_frameworks_comments.items(), 10))

    sorted_web_frameworks_views = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_web_frameworks_views = dict(islice(sorted_web_frameworks_views.items(), 10))

    answered_web_frameworks = {k: v for k, v in web_frameworks.items() if v[1] > 0}
    
    sorted_web_frameworks_response_time = dict(sorted(answered_web_frameworks.items(), key=lambda item: item[1][4]))
    top_10_web_frameworks_response_time = dict(islice(sorted_web_frameworks_response_time.items(), 10))

    sorted_web_frameworks_response_time_reverse = dict(sorted(answered_web_frameworks.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_web_frameworks_response_time_reverse = dict(islice(sorted_web_frameworks_response_time_reverse.items(), 10))

    sorted_big_data_ml_votes = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_big_data_ml_votes = dict(islice(sorted_big_data_ml_votes.items(), 10))

    sorted_big_data_ml_answers = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_big_data_ml_answers = dict(islice(sorted_big_data_ml_answers.items(), 10))

    sorted_big_data_ml_comments = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_big_data_ml_comments = dict(islice(sorted_big_data_ml_comments.items(), 10))

    sorted_big_data_ml_views = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_big_data_ml_views = dict(islice(sorted_big_data_ml_views.items(), 10))

    answered_big_data_ml = {k: v for k, v in big_data_ml.items() if v[1] > 0}

    sorted_big_data_ml_response_time = dict(sorted(answered_big_data_ml.items(), key=lambda item: item[1][4]))
    top_10_big_data_ml_response_time = dict(islice(sorted_big_data_ml_response_time.items(), 10))

    sorted_big_data_ml_response_time_reverse = dict(sorted(answered_big_data_ml.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_big_data_ml_response_time_reverse = dict(islice(sorted_big_data_ml_response_time_reverse.items(), 10))

    sorted_databases_votes = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_databases_votes = dict(islice(sorted_databases_votes.items(), 10))

    sorted_databases_answers = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_databases_answers = dict(islice(sorted_databases_answers.items(), 10))

    sorted_databases_comments = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_databases_comments = dict(islice(sorted_databases_comments.items(), 10))

    sorted_databases_views = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_databases_views = dict(islice(sorted_databases_views.items(), 10))

    answered_databases = {k: v for k, v in databases.items() if v[1] > 0}

    sorted_databases_response_time = dict(sorted(answered_databases.items(), key=lambda item: item[1][4]))
    top_10_databases_response_time = dict(islice(sorted_databases_response_time.items(), 10))

    sorted_databases_response_time_reverse = dict(sorted(answered_databases.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_databases_response_time_reverse = dict(islice(sorted_databases_response_time_reverse.items(), 10))

    sorted_platforms_votes = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_platforms_votes = dict(islice(sorted_platforms_votes.items(), 10))

    sorted_platforms_answers = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_platforms_answers = dict(islice(sorted_platforms_answers.items(), 10))

    sorted_platforms_comments = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_platforms_comments = dict(islice(sorted_platforms_comments.items(), 10))

    sorted_platforms_views = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_platforms_views = dict(islice(sorted_platforms_views.items(), 10))

    answered_platforms = {k: v for k, v in platforms.items() if v[1] > 0}

    sorted_platforms_response_time = dict(sorted(answered_platforms.items(), key=lambda item: item[1][4]))
    top_10_platforms_response_time = dict(islice(sorted_platforms_response_time.items(), 10))

    sorted_platforms_response_time_reverse = dict(sorted(answered_platforms.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_platforms_response_time_reverse = dict(islice(sorted_platforms_response_time_reverse.items(), 10))

    sorted_collaboration_tools_votes = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_collaboration_tools_votes = dict(islice(sorted_collaboration_tools_votes.items(), 10))

    sorted_collaboration_tools_answers = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_collaboration_tools_answers = dict(islice(sorted_collaboration_tools_answers.items(), 10))

    sorted_collaboration_tools_comments = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_collaboration_tools_comments = dict(islice(sorted_collaboration_tools_comments.items(), 10))

    sorted_collaboration_tools_views = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_collaboration_tools_views = dict(islice(sorted_collaboration_tools_views.items(), 10))

    answered_collaboration_tools = {k: v for k, v in collaboration_tools.items() if v[1] > 0}

    sorted_collaboration_tools_response_time = dict(sorted(answered_collaboration_tools.items(), key=lambda item: item[1][4]))
    top_10_collaboration_tools_response_time = dict(islice(sorted_collaboration_tools_response_time.items(), 10))

    sorted_collaboration_tools_response_time_reverse = dict(sorted(answered_collaboration_tools.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_collaboration_tools_response_time_reverse = dict(islice(sorted_collaboration_tools_response_time_reverse.items(), 10))

    ##############################################

    sorted_dev_tools_votes = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_dev_tools_votes = dict(islice(sorted_dev_tools_votes.items(), 10))

    sorted_dev_tools_answers = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_dev_tools_answers = dict(islice(sorted_dev_tools_answers.items(), 10))

    sorted_dev_tools_comments = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_dev_tools_comments = dict(islice(sorted_dev_tools_comments.items(), 10))

    sorted_dev_tools_views = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_dev_tools_views = dict(islice(sorted_dev_tools_views.items(), 10))

    answered_dev_tools = {k: v for k, v in dev_tools.items() if v[1] > 0}

    sorted_dev_tools_response_time = dict(sorted(answered_dev_tools.items(), key=lambda item: item[1][4]))
    top_10_dev_tools_response_time = dict(islice(sorted_dev_tools_response_time.items(), 10))

    sorted_dev_tools_response_time_reverse = dict(sorted(answered_dev_tools.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_dev_tools_response_time_reverse = dict(islice(sorted_dev_tools_response_time_reverse.items(), 10))

    distinct_users = Counter(usernames)
    sorted_distinct_users = dict(sorted(distinct_users.items(), reverse=True, key=lambda item: item[1]))
    top_10_distinct_users = dict(islice(sorted_distinct_users.items(), 10))

    sorted_ids_and_votes = dict(sorted(ids_and_votes.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_votes = dict(islice(sorted_ids_and_votes.items(), 10))

    sorted_ids_and_answers = dict(sorted(ids_and_answers.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_answers = dict(islice(sorted_ids_and_answers.items(), 10))

    sorted_ids_and_comments = dict(sorted(ids_and_comments.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_comments = dict(islice(sorted_ids_and_comments.items(), 10))

    sorted_ids_and_views = dict(sorted(ids_and_views.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_views = dict(islice(sorted_ids_and_views.items(), 10))

    answered_ids_and_response_times = {k: v for k, v in merged_dict.items() if v[2] > 0}

    sorted_ids_and_response_time = dict(sorted(answered_ids_and_response_times.items(), key=lambda item: item[1][0]))
    top_10_sorted_ids_and_response_time = dict(islice(sorted_ids_and_response_time.items(), 10))

    sorted_ids_and_response_time_reverse = dict(sorted(answered_ids_and_response_times.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_response_time_reverse = dict(islice(sorted_ids_and_response_time_reverse.items(), 10))


    numberOfComments = sum(comments)
    avgNumberOfComments = format((numberOfComments / question_number), '.3f')
    numberOfAnswers = sum(answers)
    avgNumberOfAnswers = format((numberOfAnswers / question_number), '.3f')
    numberOfVotes = sum(votes)
    avgNumberOfVotes = format((numberOfVotes / question_number), '.3f')
    yesCounter = 0
    noCounter = 0
    for snippet in code_snippets:
        if snippet == 1:
            yesCounter += 1
        else:
            noCounter += 1

    snippetData = [yesCounter, noCounter]

    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))

    for i in range(len(sorted_dates)):
        dates_and_values[sorted_dates[i]] = sorted_dates.count(sorted_dates[i])  # dict for the lineChart

    for i in range(len(tags)):
        tags_and_values[tags[i]] = tags.count(tags[i])  # dict for wordCloud

    sorted_tags_and_values = dict(
        sorted(tags_and_values.items(), reverse=True, key=lambda item: item[1]))  # sorted dict for wordCloud

    best_sorted_tags_and_values = dict(islice(sorted_tags_and_values.items(), 80))  # top 80 for wordCloud

    top_ten_tags_and_values_barchart = dict(islice(sorted_tags_and_values.items(), 10))  # top 10 for barChart
    top_twenty_tags_and_values_chord = dict(islice(sorted_tags_and_values.items(), 20))  # top 20 for chordDiagram
    top_twenty_tags = list(top_twenty_tags_and_values_chord.keys())  # to 20 tag names for the chord diagram
    all_tags_and_values = dict(sorted_tags_and_values)
    all_tags_names = list(all_tags_and_values.keys())

    for key, value in best_sorted_tags_and_values.items():  # map the dict for the wordCloud
        d = {"text": key, "size": value}
        list_of_tags_and_values.append(d)

    labels = list(dates_and_values.keys())  # lineChart labels
    values = list(dates_and_values.values())  # lineChart values

    counter = 0
    days = 0
    previous_value = 0
    halfMonthValues = []
    for key, value in dates_and_values.items():
        counter += value
        if days == 7:
            difference = abs((counter / 7) - previous_value)
            for i in range(7):
                dummy = counter / 7
                if previous_value < dummy:
                    previous_value = previous_value + (difference / 7)
                    halfMonthValues.append(previous_value)
                else:
                    previous_value = previous_value - (difference / 7)
                    halfMonthValues.append(previous_value)
            counter = 0
            days = 0
        days += 1

    added_values = values.copy()

    for i in range(1, len(added_values)):
        added_values[i] = added_values[i] + added_values[i - 1]

    barChartLabels = list(top_ten_tags_and_values_barchart.keys())  # barChart labels
    barChartValues = list(top_ten_tags_and_values_barchart.values())  # barChart values

    list_of_tuples_for_coordinates = [tuple(elem) for elem in coordinates]

    coordinates_counter_dict = dict(Counter(list_of_tuples_for_coordinates))

    coordinates_latitude = []
    coordinates_longitude = []
    coordinates_values = []

    for key, value in coordinates_counter_dict.items():
        coordinates_latitude.append(key[0])
        coordinates_longitude.append(key[1])
        coordinates_values.append(value)

    normalized_coordinates_values = [float(i) / max(coordinates_values) for i in coordinates_values]

    latLngInt = []

    for i in range(len(coordinates_values)):
        latLngInt.append([coordinates_latitude[i], coordinates_longitude[i], normalized_coordinates_values[i]])

    distinct_tags = []
    radar_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]

    stacked_open_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]
    stacked_closed_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]
    stacked_deleted_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]

    languages_tags_and_values = {}
    frameworks_tags_and_values = {}
    big_data_ml_tags_and_values = {}
    databases_tags_and_values = {}
    platforms_tags_and_values = {}
    collaboration_tools_tags_and_values = {}
    developer_tools_tags_and_values = {}

    for tag in tags:
        if tag in fields_and_techs.keys():
            if tag not in distinct_tags:
                distinct_tags.append(tag)

    for tag in distinct_tags:
        if fields_and_techs.get(tag) == 'Languages':
            radar_values[0] = radar_values[0] + 1
            languages_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Web Frameworks':
            radar_values[1] = radar_values[1] + 1
            frameworks_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Big Data - ML':
            radar_values[2] = radar_values[2] + 1
            big_data_ml_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Databases':
            radar_values[3] = radar_values[3] + 1
            databases_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Platforms':
            radar_values[4] = radar_values[4] + 1
            platforms_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Collaboration Tools':
            radar_values[5] = radar_values[5] + 1
            collaboration_tools_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Developer Tools':
            radar_values[6] = radar_values[6] + 1
            developer_tools_tags_and_values[tag] = tags.count(tag)


    sorted_languages_tags_and_values = dict(
        sorted(languages_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_frameworks_tags_and_values = dict(
        sorted(frameworks_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_big_data_ml_tags_and_values = dict(
        sorted(big_data_ml_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_databases_tags_and_values = dict(
        sorted(databases_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_platforms_tags_and_values = dict(
        sorted(platforms_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_collaboration_tools_tags_and_values = dict(
        sorted(collaboration_tools_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_developer_tools_tags_and_values = dict(
        sorted(developer_tools_tags_and_values.items(), reverse=True, key=lambda item: item[1]))  

    top_10_sorted_languages_tags_and_values = dict(islice(sorted_languages_tags_and_values.items(), 10))
    top_10_sorted_frameworks_tags_and_values = dict(islice(sorted_frameworks_tags_and_values.items(), 10))
    top_10_sorted_big_data_ml_tags_and_values = dict(islice(sorted_big_data_ml_tags_and_values.items(), 10))
    top_10_sorted_databases_tags_and_values = dict(islice(sorted_databases_tags_and_values.items(), 10))
    top_10_sorted_platforms_tags_and_values = dict(islice(sorted_platforms_tags_and_values.items(), 10))
    top_10_sorted_collaboration_tools_tags_and_values = dict(islice(sorted_collaboration_tools_tags_and_values.items(), 10))
    top_10_sorted_developer_tools_tags_and_values = dict(islice(sorted_developer_tools_tags_and_values.items(), 10))

    names_top_10_sorted_languages_tags = list(top_10_sorted_languages_tags_and_values.keys())
    names_top_10_sorted_frameworks_tags = list(top_10_sorted_frameworks_tags_and_values.keys())
    names_top_10_sorted_big_data_ml_tags = list(top_10_sorted_big_data_ml_tags_and_values.keys())
    names_top_10_sorted_databases_tags = list(top_10_sorted_databases_tags_and_values.keys())
    names_top_10_sorted_platforms_tags = list(top_10_sorted_platforms_tags_and_values.keys())
    names_top_10_sorted_collaboration_tools_tags = list(top_10_sorted_collaboration_tools_tags_and_values.keys())
    names_top_10_sorted_developer_tools_tags = list(top_10_sorted_developer_tools_tags_and_values.keys())
    

    for i in range(len(radar_values)):
        radar_values[i] = radar_values[i] / len(distinct_tags)

    # Creation of the polar chart open values.
    for question in questions:
        for tag in question['tag'].split(' '):
            if tag in fields_and_techs.keys():
                if fields_and_techs.get(tag) == 'Languages':
                    stacked_open_values[0] = stacked_open_values[0] + 1
                elif fields_and_techs.get(tag) == 'Web Frameworks':
                    stacked_open_values[1] = stacked_open_values[1] + 1
                elif fields_and_techs.get(tag) == 'Big Data - ML':
                    stacked_open_values[2] = stacked_open_values[2] + 1
                elif fields_and_techs.get(tag) == 'Databases':
                    stacked_open_values[3] = stacked_open_values[3] + 1
                elif fields_and_techs.get(tag) == 'Platforms':
                    stacked_open_values[4] = stacked_open_values[4] + 1
                elif fields_and_techs.get(tag) == 'Collaboration Tools':
                    stacked_open_values[5] = stacked_open_values[5] + 1
                elif fields_and_techs.get(tag) == 'Developer Tools':
                    stacked_open_values[6] = stacked_open_values[6] + 1
                    
    # Creation of the polar chart closed values.
    for question in closed_questions:
        for tag in question['tag'].split(' '):
            if tag in fields_and_techs.keys():
                if fields_and_techs.get(tag) == 'Languages':
                    stacked_closed_values[0] += 1
                elif fields_and_techs.get(tag) == 'Web Frameworks':
                    stacked_closed_values[1] += 1
                elif fields_and_techs.get(tag) == 'Big Data - ML':
                    stacked_closed_values[2] += 1
                elif fields_and_techs.get(tag) == 'Databases':
                    stacked_closed_values[3] += 1
                elif fields_and_techs.get(tag) == 'Platforms':
                    stacked_closed_values[4] += 1
                elif fields_and_techs.get(tag) == 'Collaboration Tools':
                    stacked_closed_values[5] += 1
                elif fields_and_techs.get(tag) == 'Developer Tools':
                    stacked_closed_values[6] += 1
    
    # Creation of the polar chart deleted values.
    for question in deleted_questions:
        for tag in question['tag'].split(' '):
            if tag in fields_and_techs.keys():
                if fields_and_techs.get(tag) == 'Languages':
                    stacked_deleted_values[0] += 1
                elif fields_and_techs.get(tag) == 'Web Frameworks':
                    stacked_deleted_values[1] += 1
                elif fields_and_techs.get(tag) == 'Big Data - ML':
                    stacked_deleted_values[2] += 1
                elif fields_and_techs.get(tag) == 'Databases':
                    stacked_deleted_values[3] += 1
                elif fields_and_techs.get(tag) == 'Platforms':
                    stacked_deleted_values[4] += 1
                elif fields_and_techs.get(tag) == 'Collaboration Tools':
                    stacked_deleted_values[5] += 1
                elif fields_and_techs.get(tag) == 'Developer Tools':
                    stacked_deleted_values[6] += 1
    
    languges_question_types = [stacked_open_values[0], stacked_deleted_values[0], stacked_closed_values[0]]
    web_frameworks_question_types = [stacked_open_values[1], stacked_deleted_values[1], stacked_closed_values[1]]
    big_data_ml_question_types = [stacked_open_values[2], stacked_deleted_values[2], stacked_closed_values[2]]
    databases_question_types = [stacked_open_values[3], stacked_deleted_values[3], stacked_closed_values[3]]
    platforms_question_types = [stacked_open_values[4], stacked_deleted_values[4], stacked_closed_values[4]]
    collaboration_tools_question_types = [stacked_open_values[5], stacked_deleted_values[5], stacked_closed_values[5]]
    dev_tools_question_types = [stacked_open_values[6], stacked_deleted_values[6], stacked_closed_values[6]]
    all_groups_question_types = [len(questions), len(deleted_questions), len(closed_questions)]


    # Distinct technologies
    distinct_technologies = []
    for tech in fields_and_techs.values():
        if tech not in distinct_technologies:
            distinct_technologies.append(tech)

    # creation of chord diagram matrix
    tag_link_matrix = np.zeros((20, 20)).astype(int)
    tags_to_be_linked = []
    languages_tag_link_matrix = np.zeros((10, 10)).astype(int)
    languages_tags_to_be_linked = []
    frameworks_tag_link_matrix = np.zeros((10, 10)).astype(int)
    frameworks_tags_to_be_linked = []
    big_data_ml_tag_link_matrix = np.zeros((10, 10)).astype(int)
    big_data_ml_tags_to_be_linked = []
    databases_tag_link_matrix = np.zeros((10, 10)).astype(int)
    databases_tags_to_be_linked = []
    platforms_tag_link_matrix = np.zeros((10, 10)).astype(int)
    platforms_tags_to_be_linked = []
    collaborations_tools_tag_link_matrix = np.zeros((10, 10)).astype(int)
    collaborations_tools_tags_to_be_linked = []
    developer_tools_tag_link_matrix = np.zeros((10, 10)).astype(int)
    developer_tools_tags_to_be_linked = []
    all_tag_link_matrix = np.zeros((len(all_tags_names), len(all_tags_names))).astype(int)
    all_tags_to_be_linked = []


    for question in questions:
        record_tags = question['tag'].split()
        if [i for i in top_twenty_tags if i in record_tags]:
            for tag in record_tags:
                if tag in top_twenty_tags:
                    tags_to_be_linked.append(top_twenty_tags.index(tag))
            if len(tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        tag_link_matrix[combination[0], combination[1]] += 1
                        tag_link_matrix[combination[1], combination[0]] += 1
            tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_languages_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_languages_tags:
                    languages_tags_to_be_linked.append(names_top_10_sorted_languages_tags.index(tag))
            if len(languages_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(languages_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        languages_tag_link_matrix[combination[0], combination[1]] += 1
                        languages_tag_link_matrix[combination[1], combination[0]] += 1
            languages_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_frameworks_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_frameworks_tags:
                    frameworks_tags_to_be_linked.append(names_top_10_sorted_frameworks_tags.index(tag))
            if len(frameworks_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(frameworks_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        frameworks_tag_link_matrix[combination[0], combination[1]] += 1
                        frameworks_tag_link_matrix[combination[1], combination[0]] += 1
            frameworks_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_big_data_ml_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_big_data_ml_tags:
                    big_data_ml_tags_to_be_linked.append(names_top_10_sorted_big_data_ml_tags.index(tag))
            if len(big_data_ml_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(big_data_ml_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        big_data_ml_tag_link_matrix[combination[0], combination[1]] += 1
                        big_data_ml_tag_link_matrix[combination[1], combination[0]] += 1
            big_data_ml_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_databases_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_databases_tags:
                    databases_tags_to_be_linked.append(names_top_10_sorted_databases_tags.index(tag))
            if len(databases_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(databases_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        databases_tag_link_matrix[combination[0], combination[1]] += 1
                        databases_tag_link_matrix[combination[1], combination[0]] += 1
            databases_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_platforms_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_platforms_tags:
                    platforms_tags_to_be_linked.append(names_top_10_sorted_platforms_tags.index(tag))
            if len(platforms_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(platforms_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        platforms_tag_link_matrix[combination[0], combination[1]] += 1
                        platforms_tag_link_matrix[combination[1], combination[0]] += 1
            platforms_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_collaboration_tools_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_collaboration_tools_tags:
                    collaborations_tools_tags_to_be_linked.append(names_top_10_sorted_collaboration_tools_tags.index(tag))
            if len(collaborations_tools_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(collaborations_tools_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        collaborations_tools_tag_link_matrix[combination[0], combination[1]] += 1
                        collaborations_tools_tag_link_matrix[combination[1], combination[0]] += 1
            collaborations_tools_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_developer_tools_tags if i in record_tags]:    
            for tag in record_tags:
                if tag in names_top_10_sorted_developer_tools_tags:
                    developer_tools_tags_to_be_linked.append(names_top_10_sorted_developer_tools_tags.index(tag))
            if len(developer_tools_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(developer_tools_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        developer_tools_tag_link_matrix[combination[0], combination[1]] += 1
                        developer_tools_tag_link_matrix[combination[1], combination[0]] += 1
            developer_tools_tags_to_be_linked.clear()
    

    list_tag_link_matrix = np.array2string(tag_link_matrix, separator=",")
    list_languages_tag_link_matrix = np.array2string(languages_tag_link_matrix, separator=",")
    list_frameworks_tag_link_matrix = np.array2string(frameworks_tag_link_matrix, separator=",")
    list_big_data_ml_tag_link_matrix = np.array2string(big_data_ml_tag_link_matrix, separator=",")
    list_databases_tag_link_matrix = np.array2string(databases_tag_link_matrix, separator=",")
    list_platforms_tag_link_matrix = np.array2string(platforms_tag_link_matrix, separator=",")
    list_collaboration_tools_tag_link_matrix = np.array2string(collaborations_tools_tag_link_matrix, separator=",")
    list_developer_tools_tag_link_matrix = np.array2string(developer_tools_tag_link_matrix, separator=",")

    inclusion_index_dict = {}
    for key,value in tag_combo_frequencies.items():
        tags = key.split(" ")
        try:
            tag_1 = int(sorted_tags_and_values[tags[0]])
            tag_2 = int(sorted_tags_and_values[tags[1]])
            tag_combo_inclusion_index = value/min(tag_1,tag_2)
            if tag_combo_inclusion_index<=1:
                inclusion_index_dict.update({key : tag_combo_inclusion_index})
        except:
            pass
        
    inclusion_index_dict = dict(sorted(inclusion_index_dict.items(), key=lambda x:x[1], reverse=True))
    top_10_inclusion_index = dict(islice(inclusion_index_dict.items(), 10))
    

    #predict_class_new_docs_dict={}

    
    return render_template('index.html', questions=questions, question_count=question_count, users=users, labels=labels,
                            values=values,
                            list_of_tags_and_values=list_of_tags_and_values, barChartLabels=barChartLabels,
                            barChartValues=barChartValues, latLngInt=latLngInt, latitudes=latitudes,
                            longitudes=longitudes,
                            distinct_technologies=distinct_technologies,
                            stacked_open_values=stacked_open_values, stacked_closed_values=stacked_closed_values,
                            stacked_deleted_values=stacked_deleted_values,
                            radar_values=radar_values, added_values=added_values, avgNumberOfAnswers=avgNumberOfAnswers,
                            avgNumberOfComments=avgNumberOfComments, avgNumberOfVotes=avgNumberOfVotes,
                            snippetData=snippetData, halfMonthValues=halfMonthValues,
                            top_10_sorted_ids_and_votes=top_10_sorted_ids_and_votes,
                            top_10_sorted_ids_and_answers=top_10_sorted_ids_and_answers,
                            top_10_sorted_ids_and_comments=top_10_sorted_ids_and_comments,
                            top_10_sorted_ids_and_views=top_10_sorted_ids_and_views,
                            top_10_distinct_users=top_10_distinct_users, location_name=location_name,
                            location_question=location_question, locations=locations,
                            top_10_languages_votes=top_10_languages_votes,
                            top_10_languages_answers=top_10_languages_answers,
                            top_10_languages_comments=top_10_languages_comments,
                            top_10_languages_views=top_10_languages_views,
                            top_10_web_frameworks_votes=top_10_web_frameworks_votes,
                            top_10_web_frameworks_answers=top_10_web_frameworks_answers,
                            top_10_web_frameworks_comments=top_10_web_frameworks_comments,
                            top_10_web_frameworks_views=top_10_web_frameworks_views,
                            top_10_big_data_ml_votes=top_10_big_data_ml_votes,
                            top_10_big_data_ml_answers=top_10_big_data_ml_answers,
                            top_10_big_data_ml_comments=top_10_big_data_ml_comments,
                            top_10_big_data_ml_views=top_10_big_data_ml_views,
                            top_10_databases_votes=top_10_databases_votes,
                            top_10_databases_answers=top_10_databases_answers,
                            top_10_databases_comments=top_10_databases_comments,
                            top_10_databases_views=top_10_databases_views,
                            top_10_platforms_votes=top_10_platforms_votes,
                            top_10_platforms_answers=top_10_platforms_answers,
                            top_10_platforms_comments=top_10_platforms_comments,
                            top_10_platforms_views=top_10_platforms_views,
                            top_10_collaboration_tools_votes=top_10_collaboration_tools_votes,
                            top_10_collaboration_tools_answers=top_10_collaboration_tools_answers,
                            top_10_collaboration_tools_comments=top_10_collaboration_tools_comments,
                            top_10_collaboration_tools_views=top_10_collaboration_tools_views,
                            top_10_dev_tools_votes=top_10_dev_tools_votes,
                            top_10_dev_tools_answers=top_10_dev_tools_answers,
                            top_10_dev_tools_comments=top_10_dev_tools_comments,
                            top_10_dev_tools_views=top_10_dev_tools_views,
                            date_from=date_from, date_to=date_to, 
                            list_tag_link_matrix=list_tag_link_matrix, top_twenty_tags=top_twenty_tags, 
                            list_languages_tag_link_matrix = list_languages_tag_link_matrix, names_top_10_sorted_languages_tags = names_top_10_sorted_languages_tags,
                            list_frameworks_tag_link_matrix = list_frameworks_tag_link_matrix, names_top_10_sorted_frameworks_tags = names_top_10_sorted_frameworks_tags,
                            list_big_data_ml_tag_link_matrix = list_big_data_ml_tag_link_matrix, names_top_10_sorted_big_data_ml_tags = names_top_10_sorted_big_data_ml_tags,
                            list_databases_tag_link_matrix = list_databases_tag_link_matrix, names_top_10_sorted_databases_tags = names_top_10_sorted_databases_tags,
                            list_platforms_tag_link_matrix = list_platforms_tag_link_matrix, names_top_10_sorted_platforms_tags = names_top_10_sorted_platforms_tags,
                            list_collaboration_tools_tag_link_matrix = list_collaboration_tools_tag_link_matrix, names_top_10_sorted_collaboration_tools_tags = names_top_10_sorted_collaboration_tools_tags,
                            list_developer_tools_tag_link_matrix = list_developer_tools_tag_link_matrix, names_top_10_sorted_developer_tools_tags = names_top_10_sorted_developer_tools_tags,
                            times_data_list = times_data_list, survival_time_curve_values = survival_time_curve_values, 
                            languages_times_data_list = languages_times_data_list, languages_survival_time_curve_values = languages_survival_time_curve_values,
                            web_frameworks_times_data_list = web_frameworks_times_data_list, web_frameworks_survival_time_curve_values = web_frameworks_survival_time_curve_values,
                            big_data_ml_times_data_list = big_data_ml_times_data_list, big_data_ml_survival_time_curve_values = big_data_ml_survival_time_curve_values,
                            databases_times_data_list = databases_times_data_list, databases_survival_time_curve_values = databases_survival_time_curve_values,
                            platforms_times_data_list = platforms_times_data_list, platforms_survival_time_curve_values = platforms_survival_time_curve_values,
                            collaboration_tools_times_data_list = collaboration_tools_times_data_list, collaboration_tools_survival_time_curve_values = collaboration_tools_survival_time_curve_values,
                            dev_tools_times_data_list = dev_tools_times_data_list, dev_tools_survival_time_curve_values = dev_tools_survival_time_curve_values,
                            languges_question_types = languges_question_types,
                            web_frameworks_question_types = web_frameworks_question_types,
                            big_data_ml_question_types = big_data_ml_question_types,
                            databases_question_types = databases_question_types,
                            platforms_question_types = platforms_question_types,
                            collaboration_tools_question_types = collaboration_tools_question_types,
                            dev_tools_question_types = dev_tools_question_types,
                            all_groups_question_types = all_groups_question_types,
                            sorted_comments_distribution_labels = sorted_comments_distribution_labels,
                            sorted_comments_distribution_values = sorted_comments_distribution_values,
                            sorted_answers_distribution_labels = sorted_answers_distribution_labels,
                            sorted_answers_distribution_values = sorted_answers_distribution_values,
                            sorted_votes_distribution_labels = sorted_votes_distribution_labels,
                            sorted_votes_distribution_values = sorted_votes_distribution_values,
                            sorted_views_distribution_labels = sorted_views_distribution_labels,
                            sorted_views_distribution_values = sorted_views_distribution_values,
                            answeredData = answeredData, top_10_inclusion_index = top_10_inclusion_index,
                            median_times_dict = median_times_dict,
                            top_10_languages_response_time = top_10_languages_response_time,
                            top_10_web_frameworks_response_time = top_10_web_frameworks_response_time,
                            top_10_big_data_ml_response_time = top_10_big_data_ml_response_time,
                            top_10_databases_response_time = top_10_databases_response_time,
                            top_10_platforms_response_time = top_10_platforms_response_time,
                            top_10_collaboration_tools_response_time = top_10_collaboration_tools_response_time,
                            top_10_dev_tools_response_time = top_10_dev_tools_response_time,
                            top_10_sorted_ids_and_response_time = top_10_sorted_ids_and_response_time,
                            top_10_languages_response_time_reverse=top_10_languages_response_time_reverse,
                            top_10_web_frameworks_response_time_reverse=top_10_web_frameworks_response_time_reverse,
                            top_10_big_data_ml_response_time_reverse=top_10_big_data_ml_response_time_reverse,
                            top_10_databases_response_time_reverse=top_10_databases_response_time_reverse,
                            top_10_platforms_response_time_reverse=top_10_platforms_response_time_reverse,
                            top_10_collaboration_tools_response_time_reverse=top_10_collaboration_tools_response_time_reverse,
                            top_10_dev_tools_response_time_reverse=top_10_dev_tools_response_time_reverse,
                            top_10_sorted_ids_and_response_time_reverse=top_10_sorted_ids_and_response_time_reverse,
                            predict_class_new_docs_dict = {},res_dif_0_formatted = {}, res_dif_1_formatted = {}, res_growth_dict={}, ptc_dict = {}
                           )
@app.route('/get_bert')
def get_map():
    return render_template('topic_visualization/bert_visualize_docs_reduced.html')

@app.route('/get_bert2')
def get_map2():
    return render_template('topic_visualization/bert_visualize_hierarchy_reduced.html')

@app.route('/get_bert3')
def get_map3():
    return render_template('topic_visualization/bert_visualize_reduced.html')

@app.route('/get_ap_ii')
def get_map4():
    return render_template('topic_visualization/ap_ii_network.html')

@app.route('/get_dates', methods=['GET'])
def fetch():
    client = MongoClient('localhost', 27017)
    db = client['LLM-db']
    covidCollection = db.questions
    date_from = request.args.get('dateFrom')
    date_to = request.args.get('dateTo')
    closed = int(request.args.get('inclClosed')) if request.args.get('inclClosed') != None else 0
    if closed == 1: # all questions
        query = {'timestamps': {'$gte': date_from, '$lte': date_to}}
    elif closed == 0: # only open questions
        query = {
            "$and": [
                {"timestamps": {"$gte": date_from, "$lte": date_to}},
                {"$and": [
                    {"closed": closed},
                    {"deleted": closed}
                ]}
            ]
        }
    
    questions = covidCollection.find(query)  #load questions from collection
    
    techCollection = db.technologies_list
    technologies = techCollection.find()
    users = len(questions.distinct('owner_id'))
    question_number = len(covidCollection.distinct('_id'))
    questions = list(questions)
    
    all_questions = covidCollection.find({'timestamps': {'$gte': date_from, '$lte': date_to}})
    all_questions_list = list(all_questions)
    closed_questions = list([q for q in all_questions_list if q['closed'] == 1])
    deleted_questions = list([q for q in all_questions_list if q['deleted'] == 1])

    all_filtered_questions = covidCollection.find({
            "$and": [
                {"timestamps": {"$gte": date_from, "$lte": date_to}},
                {"$and": [
                    {"closed": 0},
                    {"deleted": 0}
                ]}
            ]
        })
    all_filtered_questions_list = list(all_filtered_questions)
    
    
    dates = []
    dates_and_values = {}
    tags = []
    tags_and_values = {}
    list_of_tags_and_values = []
    latitudes = []
    longitudes = []
    coordinates = []
    comments = []
    answers = []
    votes = []
    code_snippets = []
    ids_and_votes = {}
    ids_and_answers = {}
    ids_and_comments = {}
    ids_and_views = {}
    ids_and_response_times = {}
    usernames = []
    locations = []
    location_name = []
    location_question = []
    languages = {}
    web_frameworks = {}
    big_data_ml = {}
    databases = {}
    platforms = {}
    collaboration_tools = {}
    dev_tools = {}
    elapsed_time_data_list = []
    languages_elapsed_time_data_list = []
    web_frameworks_elapsed_time_data_list = []
    big_data_ml_elapsed_time_data_list = []
    databases_elapsed_time_data_list = []
    platforms_elapsed_time_data_list = []
    collaboration_tools_elapsed_time_data_list = []
    dev_tools_elapsed_time_data_list = []
    comments_distribution = {}
    votes_distribution = {}
    answers_distribution = {}
    views_distribution = {}
    tag_combo_frequencies = {}
    answered_questions = 0
    unanswered_questions = 0

    fields_and_techs = {}
    for technology in technologies:
        fields_and_techs.update({technology['field'].lower(): technology['technology']})

    question_count = 0
    for question in questions:
        question_count += 1
        dates.append(question['timestamps'][:10])
        record_tags = question['tag'].split()
        dif_tags = list(set(record_tags))

        if question['owner_id'] != 'No Owner ID':
            usernames.append(question['owner_id'])

        comments.append(question['comments'])
        if question['comments'] in comments_distribution.keys():
            comments_distribution[question['comments']]+=1
        else:
            comments_distribution[int(question['comments'])] =1

        answers.append(int(question['answers']))
        if question['answers'] in answers_distribution.keys():
            answers_distribution[int(question['answers'])] += 1
        else:
            answers_distribution[int(question['answers'])] = 1
        if int(question['answers']) > 0:
            answered_questions += 1
        else:
            unanswered_questions += 1

        integer_votes = int(question['votes'])
        votes.append(integer_votes)
        if integer_votes in votes_distribution.keys():
            votes_distribution[integer_votes] += 1
        else:
            votes_distribution[integer_votes] = 1

        views_text = str((question['views'])).replace(",", "")
        views = re.findall('[0-9]+', views_text)

        try:
            if question['tag_combinations'] != 'No tag combinations':
                if len(question['tag_combinations']) < 1000:
                    for tag_combo in question['tag_combinations']:
                        tag_combo_string = ' '.join(tag_combo)
                        if not tag_combo_string in tag_combo_frequencies:
                            tag_combo_frequencies[tag_combo_string] = 1
                        else:
                            tag_combo_frequencies[tag_combo_string] += 1
        except:
            pass

        if len(views) != 0:
            views_integer = int(views[0])
            if views_integer in views_distribution.keys():
                views_distribution[views_integer] += 1
            else:
                views_distribution.update({views_integer : 1})

        q_id = question['question_id']
        q_link = "https://stackoverflow.com/questions/" + str(re.sub("[^0-9]", "", q_id))
        ids_and_votes.update({q_link: [int(question['votes']), question['question_title']]})
        ids_and_answers.update({q_link: [int(question['answers']), question['question_title']]})
        ids_and_comments.update({q_link: [int(question['comments']), question['question_title']]})
        ids_and_views.update({q_link: [views_integer, question['question_title']]})
        #########################
        
        question_time = datetime.fromisoformat(question['timestamps'][:-1]).replace(tzinfo=timezone.utc)
        date_to_formatted = datetime.fromisoformat(date_to + " 23:59:59").replace(tzinfo=timezone.utc)
        if question_time:           
            if question['first_answer'] != 'No answers':
                first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                if first_answer_time <= date_to_formatted:
                    hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                    event = 1
                else:      
                    hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                    event = 0   
            else:      
                hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                event = 0    
            if hour_diff!=0.0:
                elapsed_time_data_list.append([hour_diff,event])            
        
        ids_and_response_times.update({q_link: [hour_diff, question['question_title']]})
        merged_dict = {}
        for key in ids_and_response_times.keys():
            merged_dict[key] = [ids_and_response_times[key][0], ids_and_response_times[key][1], ids_and_answers[key][0]]
        #########################
        for q_tag in dif_tags:
            if fields_and_techs.get(q_tag) == 'Languages':
                languages.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        if first_answer_time <= date_to_formatted:
                            hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                            event = 1
                        else:      
                            hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                            event = 0    
                    else:
                        hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        languages_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Web Frameworks':
                web_frameworks.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        if first_answer_time <= date_to_formatted:
                            hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                            event = 1
                        else:      
                            hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                            event = 0  
                    else:
                        hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        web_frameworks_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Big Data - ML':
                big_data_ml.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        if first_answer_time <= date_to_formatted:
                            hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                            event = 1
                        else:      
                            hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                            event = 0  
                    else:
                        hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        big_data_ml_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Databases':
                databases.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        if first_answer_time <= date_to_formatted:
                            hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                            event = 1
                        else:      
                            hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                            event = 0 
                    else:
                        hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        databases_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Platforms':
                platforms.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        if first_answer_time <= date_to_formatted:
                            hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                            event = 1
                        else:      
                            hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                            event = 0  
                    else:
                        hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        platforms_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Collaboration Tools':
                collaboration_tools.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        if first_answer_time <= date_to_formatted:
                            hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                            event = 1
                        else:      
                            hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                            event = 0  
                    else:
                        hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        collaboration_tools_elapsed_time_data_list.append([hour_diff,event])
            if fields_and_techs.get(q_tag) == 'Developer Tools':
                dev_tools.update({q_link: [int(question['votes']), int(question['answers']), int(question['comments']), views_integer, hour_diff, question['question_title']]})
                if question_time:           
                    if question['first_answer'] != 'No answers':
                        first_answer_time = datetime.fromisoformat(question['first_answer'][:-1]).replace(tzinfo=timezone.utc)
                        if first_answer_time <= date_to_formatted:
                            hour_diff = round((first_answer_time - question_time).total_seconds() / 60,1)
                            event = 1
                        else:      
                            hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                            event = 0   
                    else:
                        hour_diff = round((date_to_formatted - question_time).total_seconds() / 60,1)
                        event = 0    
                    if hour_diff!=0.0:
                        dev_tools_elapsed_time_data_list.append([hour_diff,event])
        #########################
        code_snippets.append(int(question['code_snippet']))
        if question['latitude'] != 'None':
            latLng = [question['latitude'], question['longitude']]
            coordinates.append(latLng)
            latitudes.append(question['latitude'])
            longitudes.append(question['longitude'])
            locations.append(question['location'])
        for i in range(len(record_tags)):
            tags.append(record_tags[i])
    
    answeredData = [answered_questions, unanswered_questions]
                
    sorted_comments_distribution = dict(sorted(comments_distribution.items(), key=lambda x:x[0]))
    sorted_comments_distribution_labels = list(sorted_comments_distribution.keys())
    sorted_comments_distribution_values = list(sorted_comments_distribution.values())

    sorted_answers_distribution = dict(sorted(answers_distribution.items(), key=lambda x:x[0]))
    sorted_answers_distribution_labels = list(sorted_answers_distribution.keys())
    sorted_answers_distribution_values = list(sorted_answers_distribution.values())
    
    sorted_votes_distribution = dict(sorted(votes_distribution.items(), key=lambda x:x[0]))
    sorted_votes_distribution_labels = list(sorted_votes_distribution.keys())
    sorted_votes_distribution_values = list(sorted_votes_distribution.values())
    
    sorted_views_distribution = dict(sorted(views_distribution.items(), key=lambda x:x[0]))
    sorted_views_distribution_labels = list(sorted_views_distribution.keys())
    sorted_views_distribution_values = list(sorted_views_distribution.values())

    
    sorted_elapsed_time_data_list = sorted(elapsed_time_data_list, key=itemgetter(0))
    languages_sorted_elapsed_time_data_list = sorted(languages_elapsed_time_data_list, key=itemgetter(0))
    web_frameworks_sorted_elapsed_time_data_list = sorted(web_frameworks_elapsed_time_data_list, key=itemgetter(0))
    big_data_ml_sorted_elapsed_time_data_list = sorted(big_data_ml_elapsed_time_data_list, key=itemgetter(0))
    databases_sorted_elapsed_time_data_list = sorted(databases_elapsed_time_data_list, key=itemgetter(0))
    platforms_sorted_elapsed_time_data_list = sorted(platforms_elapsed_time_data_list, key=itemgetter(0))
    collaboration_tools_sorted_elapsed_time_data_list = sorted(collaboration_tools_elapsed_time_data_list, key=itemgetter(0))
    dev_tools_sorted_elapsed_time_data_list = sorted(dev_tools_elapsed_time_data_list, key=itemgetter(0))
    
    elapsed_times_all = []
    elapsed_times_languages = []
    elapsed_times_web_frameworks = []
    elapsed_times_big_data_ml = []
    elapsed_times_databases = []
    elapsed_times_platforms = []
    elapsed_times_colab_tools = []
    elapsed_times_dev_tools = []
    
    for time_list in sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_all.append(time_list[0])
    
    for time_list in languages_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_languages.append(time_list[0])
    
    for time_list in web_frameworks_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_web_frameworks.append(time_list[0])
    
    for time_list in big_data_ml_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_big_data_ml.append(time_list[0])        
    
    for time_list in databases_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_databases.append(time_list[0])
    
    for time_list in platforms_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_platforms.append(time_list[0])
    
    for time_list in collaboration_tools_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_colab_tools.append(time_list[0])
            
    for time_list in dev_tools_sorted_elapsed_time_data_list:
        if time_list[1] == 1:
            elapsed_times_dev_tools.append(time_list[0])
    
    median_times_dict = {
        "All" : median(elapsed_times_all) if len(elapsed_times_all) != 0 else 0,
        "Languages" : median(elapsed_times_languages) if len(elapsed_times_languages) != 0 else 0,
        "Web Frameworks" : median(elapsed_times_web_frameworks) if len(elapsed_times_web_frameworks) != 0 else 0,
        "Big Data - ML" : median(elapsed_times_big_data_ml) if len(elapsed_times_big_data_ml) != 0 else 0,
        "Databases" : median(elapsed_times_databases) if len(elapsed_times_databases) != 0 else 0,
        "Platforms" : median(elapsed_times_platforms) if len(elapsed_times_platforms) != 0 else 0,
        "Collaboration Tools" : median(elapsed_times_colab_tools) if len(elapsed_times_colab_tools) != 0 else 0,
        "Developer Tools" : median(elapsed_times_dev_tools) if len(elapsed_times_dev_tools) != 0 else 0
    }
    
    
    times_data_list = []
    number_of_distinct_times_data_list = []
    censored_data_list = []
    times_left_list = []
    
    item_counter = 1
    times_data_list.append(0)
    number_of_distinct_times_data_list.append(0)
    censored_data_list.append(0)
    times_left_list.append(len(sorted_elapsed_time_data_list))
    
    for item in sorted_elapsed_time_data_list:
        if item[0] not in times_data_list :
            if item[1] == 1:
                times_data_list.append(item[0])
                number_of_distinct_times_data_list.append(1)
                censored_data_list.append(0) 
                times_left_list.append(times_left_list[item_counter - 1] - (number_of_distinct_times_data_list[item_counter - 1] + censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                censored_data_list[item_counter-1]+=1
        
    survival_time_curve_values = []
    for i in range(len(times_data_list)):
        if i == 0:
            survival_time_curve_values.append((times_left_list[i] - number_of_distinct_times_data_list[i]) /  times_left_list[i]) if times_left_list[i] != 0 else 0
        else:
            survival_time_curve_values.append(((times_left_list[i] - number_of_distinct_times_data_list[i]) /  times_left_list[i])* survival_time_curve_values[i-1]) if times_left_list[i] != 0 else 0
    
    
    
    languages_times_data_list = []
    languages_number_of_distinct_times_data_list = []
    languages_censored_data_list = []
    languages_times_left_list = []

    item_counter = 1
    languages_times_data_list.append(0)
    languages_number_of_distinct_times_data_list.append(0)
    languages_censored_data_list.append(0)
    languages_times_left_list.append(len(languages_sorted_elapsed_time_data_list))

    for item in languages_sorted_elapsed_time_data_list:
        if item[0] not in languages_times_data_list :
            if item[1] == 1:
                languages_times_data_list.append(item[0])
                languages_number_of_distinct_times_data_list.append(1)
                languages_censored_data_list.append(0) 
                languages_times_left_list.append(languages_times_left_list[item_counter - 1] - (languages_number_of_distinct_times_data_list[item_counter - 1] + languages_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                languages_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                languages_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                languages_censored_data_list[item_counter-1]+=1

    languages_survival_time_curve_values = []
    for i in range(len(languages_times_data_list)):
        if i == 0:
            languages_survival_time_curve_values.append((languages_times_left_list[i] - languages_number_of_distinct_times_data_list[i]) /  languages_times_left_list[i]) if languages_times_left_list[i] != 0 else 0
        else:
            languages_survival_time_curve_values.append(((languages_times_left_list[i] - languages_number_of_distinct_times_data_list[i]) /  languages_times_left_list[i]) * languages_survival_time_curve_values[i-1]) if languages_times_left_list[i] != 0 else 0
        
    
    
    web_frameworks_times_data_list = []
    web_frameworks_number_of_distinct_times_data_list = []
    web_frameworks_censored_data_list = []
    web_frameworks_times_left_list = []

    item_counter = 1
    web_frameworks_times_data_list.append(0)
    web_frameworks_number_of_distinct_times_data_list.append(0)
    web_frameworks_censored_data_list.append(0)
    web_frameworks_times_left_list.append(len(web_frameworks_sorted_elapsed_time_data_list))

    for item in web_frameworks_sorted_elapsed_time_data_list:
        if item[0] not in web_frameworks_times_data_list :
            if item[1] == 1:
                web_frameworks_times_data_list.append(item[0])
                web_frameworks_number_of_distinct_times_data_list.append(1)
                web_frameworks_censored_data_list.append(0) 
                web_frameworks_times_left_list.append(web_frameworks_times_left_list[item_counter - 1] - (web_frameworks_number_of_distinct_times_data_list[item_counter - 1] + web_frameworks_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                web_frameworks_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                web_frameworks_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                web_frameworks_censored_data_list[item_counter-1]+=1
                
    web_frameworks_survival_time_curve_values = []
    for i in range(len(web_frameworks_times_data_list)):
        if i == 0:
            web_frameworks_survival_time_curve_values.append((web_frameworks_times_left_list[i] - web_frameworks_number_of_distinct_times_data_list[i]) /  web_frameworks_times_left_list[i]) if web_frameworks_times_left_list[i] != 0 else 0
        else:
            web_frameworks_survival_time_curve_values.append(((web_frameworks_times_left_list[i] - web_frameworks_number_of_distinct_times_data_list[i]) /  web_frameworks_times_left_list[i])* web_frameworks_survival_time_curve_values[i-1]) if web_frameworks_times_left_list[i] != 0 else 0
        
        

    big_data_ml_times_data_list = []
    big_data_ml_number_of_distinct_times_data_list = []
    big_data_ml_censored_data_list = []
    big_data_ml_times_left_list = []

    item_counter = 1
    big_data_ml_times_data_list.append(0)
    big_data_ml_number_of_distinct_times_data_list.append(0)
    big_data_ml_censored_data_list.append(0)
    big_data_ml_times_left_list.append(len(big_data_ml_sorted_elapsed_time_data_list))

    for item in big_data_ml_sorted_elapsed_time_data_list:
        if item[0] not in big_data_ml_times_data_list :
            if item[1] == 1:
                big_data_ml_times_data_list.append(item[0])
                big_data_ml_number_of_distinct_times_data_list.append(1)
                big_data_ml_censored_data_list.append(0) 
                big_data_ml_times_left_list.append(big_data_ml_times_left_list[item_counter - 1] - (big_data_ml_number_of_distinct_times_data_list[item_counter - 1] + big_data_ml_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                big_data_ml_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                big_data_ml_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                big_data_ml_censored_data_list[item_counter-1]+=1
                
    big_data_ml_survival_time_curve_values = []
    for i in range(len(big_data_ml_times_data_list)):
        if i == 0:
            big_data_ml_survival_time_curve_values.append((big_data_ml_times_left_list[i] - big_data_ml_number_of_distinct_times_data_list[i]) /  big_data_ml_times_left_list[i]) if big_data_ml_times_left_list[i] != 0 else 0
        else:
            big_data_ml_survival_time_curve_values.append(((big_data_ml_times_left_list[i] - big_data_ml_number_of_distinct_times_data_list[i]) /  big_data_ml_times_left_list[i])* big_data_ml_survival_time_curve_values[i-1]) if big_data_ml_times_left_list[i] != 0 else 0
            
    databases_times_data_list = []
    databases_number_of_distinct_times_data_list = []
    databases_censored_data_list = []
    databases_times_left_list = []

    item_counter = 1
    databases_times_data_list.append(0)
    databases_number_of_distinct_times_data_list.append(0)
    databases_censored_data_list.append(0)
    databases_times_left_list.append(len(databases_sorted_elapsed_time_data_list))

    for item in databases_sorted_elapsed_time_data_list:
        if item[0] not in databases_times_data_list :
            if item[1] == 1:
                databases_times_data_list.append(item[0])
                databases_number_of_distinct_times_data_list.append(1)
                databases_censored_data_list.append(0) 
                databases_times_left_list.append(databases_times_left_list[item_counter - 1] - (databases_number_of_distinct_times_data_list[item_counter - 1] + databases_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                databases_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                databases_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                databases_censored_data_list[item_counter-1]+=1
                
    databases_survival_time_curve_values = []
    for i in range(len(databases_times_data_list)):
        if i == 0:
            databases_survival_time_curve_values.append((databases_times_left_list[i] - databases_number_of_distinct_times_data_list[i]) /  databases_times_left_list[i]) if databases_times_left_list[i] != 0 else 0
        else:
            databases_survival_time_curve_values.append(((databases_times_left_list[i] - databases_number_of_distinct_times_data_list[i]) /  databases_times_left_list[i])* databases_survival_time_curve_values[i-1]) if databases_times_left_list[i] != 0 else 0
            
    
    platforms_times_data_list = []
    platforms_number_of_distinct_times_data_list = []
    platforms_censored_data_list = []
    platforms_times_left_list = []

    item_counter = 1
    platforms_times_data_list.append(0)
    platforms_number_of_distinct_times_data_list.append(0)
    platforms_censored_data_list.append(0)
    platforms_times_left_list.append(len(platforms_sorted_elapsed_time_data_list))

    for item in platforms_sorted_elapsed_time_data_list:
        if item[0] not in platforms_times_data_list :
            if item[1] == 1:
                platforms_times_data_list.append(item[0])
                platforms_number_of_distinct_times_data_list.append(1)
                platforms_censored_data_list.append(0) 
                platforms_times_left_list.append(platforms_times_left_list[item_counter - 1] - (platforms_number_of_distinct_times_data_list[item_counter - 1] + platforms_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                platforms_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                platforms_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                platforms_censored_data_list[item_counter-1]+=1
                
    platforms_survival_time_curve_values = []
    for i in range(len(platforms_times_data_list)):
        if i == 0:
            platforms_survival_time_curve_values.append((platforms_times_left_list[i] - platforms_number_of_distinct_times_data_list[i]) /  platforms_times_left_list[i]) if platforms_times_left_list[i] != 0 else 0
        else:
            platforms_survival_time_curve_values.append(((platforms_times_left_list[i] - platforms_number_of_distinct_times_data_list[i]) /  platforms_times_left_list[i])* platforms_survival_time_curve_values[i-1]) if platforms_times_left_list[i] != 0 else 0
    
    
    
    collaboration_tools_times_data_list = []
    collaboration_tools_number_of_distinct_times_data_list = []
    collaboration_tools_censored_data_list = []
    collaboration_tools_times_left_list = []

    item_counter = 1
    collaboration_tools_times_data_list.append(0)
    collaboration_tools_number_of_distinct_times_data_list.append(0)
    collaboration_tools_censored_data_list.append(0)
    collaboration_tools_times_left_list.append(len(collaboration_tools_sorted_elapsed_time_data_list))

    for item in collaboration_tools_sorted_elapsed_time_data_list:
        if item[0] not in collaboration_tools_times_data_list :
            if item[1] == 1:
                collaboration_tools_times_data_list.append(item[0])
                collaboration_tools_number_of_distinct_times_data_list.append(1)
                collaboration_tools_censored_data_list.append(0) 
                collaboration_tools_times_left_list.append(collaboration_tools_times_left_list[item_counter - 1] - (collaboration_tools_number_of_distinct_times_data_list[item_counter - 1] + collaboration_tools_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                collaboration_tools_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                collaboration_tools_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                collaboration_tools_censored_data_list[item_counter-1]+=1
                
    collaboration_tools_survival_time_curve_values = []
    for i in range(len(collaboration_tools_times_data_list)):
        if i == 0:
            collaboration_tools_survival_time_curve_values.append((collaboration_tools_times_left_list[i] - collaboration_tools_number_of_distinct_times_data_list[i]) /  collaboration_tools_times_left_list[i]) if collaboration_tools_times_left_list[i] != 0 else 0
        else:
            collaboration_tools_survival_time_curve_values.append(((collaboration_tools_times_left_list[i] - collaboration_tools_number_of_distinct_times_data_list[i]) /  collaboration_tools_times_left_list[i])* collaboration_tools_survival_time_curve_values[i-1]) if collaboration_tools_times_left_list[i] != 0 else 0
        
        
    
    dev_tools_times_data_list = []
    dev_tools_number_of_distinct_times_data_list = []
    dev_tools_censored_data_list = []
    dev_tools_times_left_list = []

    item_counter = 1
    dev_tools_times_data_list.append(0)
    dev_tools_number_of_distinct_times_data_list.append(0)
    dev_tools_censored_data_list.append(0)
    dev_tools_times_left_list.append(len(dev_tools_sorted_elapsed_time_data_list))

    for item in dev_tools_sorted_elapsed_time_data_list:
        if item[0] not in dev_tools_times_data_list :
            if item[1] == 1:
                dev_tools_times_data_list.append(item[0])
                dev_tools_number_of_distinct_times_data_list.append(1)
                dev_tools_censored_data_list.append(0) 
                dev_tools_times_left_list.append(dev_tools_times_left_list[item_counter - 1] - (dev_tools_number_of_distinct_times_data_list[item_counter - 1] + dev_tools_censored_data_list[item_counter-1]))  
                item_counter+=1        
            else:
                dev_tools_censored_data_list[item_counter-1]+=1
            
        else:         
            if item[1] == 1:
                dev_tools_number_of_distinct_times_data_list[item_counter - 1]+=1
            else:
                dev_tools_censored_data_list[item_counter-1]+=1
                
    dev_tools_survival_time_curve_values = []
    for i in range(len(dev_tools_times_data_list)):
        if i == 0:
            dev_tools_survival_time_curve_values.append((dev_tools_times_left_list[i] - dev_tools_number_of_distinct_times_data_list[i]) /  dev_tools_times_left_list[i]) if dev_tools_times_left_list[i] != 0 else 0
        else:
            dev_tools_survival_time_curve_values.append(((dev_tools_times_left_list[i] - dev_tools_number_of_distinct_times_data_list[i]) /  dev_tools_times_left_list[i])* dev_tools_survival_time_curve_values[i-1]) if dev_tools_times_left_list[i] != 0 else 0    
        
    
             
    distinct_locations = Counter(locations)
    for key, value in distinct_locations.items():
        location_name.append(key)
        location_question.append(value)

    distinct_locations = Counter(locations)
    for key, value in distinct_locations.items():
        location_name.append(key)
        location_question.append(value)

    sorted_language_votes = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_languages_votes = dict(islice(sorted_language_votes.items(), 10))

    sorted_language_answers = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_languages_answers = dict(islice(sorted_language_answers.items(), 10))

    sorted_language_comments = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_languages_comments = dict(islice(sorted_language_comments.items(), 10))

    sorted_language_views = dict(sorted(languages.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_languages_views = dict(islice(sorted_language_views.items(), 10))

    answered_languages = {k: v for k, v in languages.items() if v[1] > 0}

    sorted_language_response_time = dict(sorted(answered_languages.items(), key=lambda item: item[1][4]))
    top_10_languages_response_time = dict(islice(sorted_language_response_time.items(), 10))

    sorted_language_response_time_reverse = dict(sorted(answered_languages.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_languages_response_time_reverse = dict(islice(sorted_language_response_time_reverse.items(), 10))

    sorted_web_frameworks_votes = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_web_frameworks_votes = dict(islice(sorted_web_frameworks_votes.items(), 10))

    sorted_web_frameworks_answers = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_web_frameworks_answers = dict(islice(sorted_web_frameworks_answers.items(), 10))

    sorted_web_frameworks_comments = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_web_frameworks_comments = dict(islice(sorted_web_frameworks_comments.items(), 10))

    sorted_web_frameworks_views = dict(sorted(web_frameworks.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_web_frameworks_views = dict(islice(sorted_web_frameworks_views.items(), 10))

    answered_web_frameworks = {k: v for k, v in web_frameworks.items() if v[1] > 0}

    sorted_web_frameworks_response_time = dict(sorted(answered_web_frameworks.items(), key=lambda item: item[1][4]))
    top_10_web_frameworks_response_time = dict(islice(sorted_web_frameworks_response_time.items(), 10))

    sorted_web_frameworks_response_time_reverse = dict(sorted(answered_web_frameworks.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_web_frameworks_response_time_reverse = dict(islice(sorted_web_frameworks_response_time_reverse.items(), 10))

    sorted_big_data_ml_votes = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_big_data_ml_votes = dict(islice(sorted_big_data_ml_votes.items(), 10))

    sorted_big_data_ml_answers = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_big_data_ml_answers = dict(islice(sorted_big_data_ml_answers.items(), 10))

    sorted_big_data_ml_comments = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_big_data_ml_comments = dict(islice(sorted_big_data_ml_comments.items(), 10))

    sorted_big_data_ml_views = dict(sorted(big_data_ml.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_big_data_ml_views = dict(islice(sorted_big_data_ml_views.items(), 10))

    answered_big_data_ml = {k: v for k, v in big_data_ml.items() if v[1] > 0}

    sorted_big_data_ml_response_time = dict(sorted(answered_big_data_ml.items(), key=lambda item: item[1][4]))
    top_10_big_data_ml_response_time = dict(islice(sorted_big_data_ml_response_time.items(), 10))

    sorted_big_data_ml_response_time_reverse = dict(sorted(answered_big_data_ml.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_big_data_ml_response_time_reverse = dict(islice(sorted_big_data_ml_response_time_reverse.items(), 10))

    sorted_databases_votes = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_databases_votes = dict(islice(sorted_databases_votes.items(), 10))

    sorted_databases_answers = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_databases_answers = dict(islice(sorted_databases_answers.items(), 10))

    sorted_databases_comments = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_databases_comments = dict(islice(sorted_databases_comments.items(), 10))

    sorted_databases_views = dict(sorted(databases.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_databases_views = dict(islice(sorted_databases_views.items(), 10))

    answered_databases = {k: v for k, v in databases.items() if v[1] > 0}

    sorted_databases_response_time = dict(sorted(answered_databases.items(), key=lambda item: item[1][4]))
    top_10_databases_response_time = dict(islice(sorted_databases_response_time.items(), 10))

    sorted_databases_response_time_reverse = dict(sorted(answered_databases.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_databases_response_time_reverse = dict(islice(sorted_databases_response_time_reverse.items(), 10))

    sorted_platforms_votes = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_platforms_votes = dict(islice(sorted_platforms_votes.items(), 10))

    sorted_platforms_answers = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_platforms_answers = dict(islice(sorted_platforms_answers.items(), 10))

    sorted_platforms_comments = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_platforms_comments = dict(islice(sorted_platforms_comments.items(), 10))

    sorted_platforms_views = dict(sorted(platforms.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_platforms_views = dict(islice(sorted_platforms_views.items(), 10))

    answered_platforms = {k: v for k, v in platforms.items() if v[1] > 0}

    sorted_platforms_response_time = dict(sorted(answered_platforms.items(), key=lambda item: item[1][4]))
    top_10_platforms_response_time = dict(islice(sorted_platforms_response_time.items(), 10))

    sorted_platforms_response_time_reverse = dict(sorted(answered_platforms.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_platforms_response_time_reverse = dict(islice(sorted_platforms_response_time_reverse.items(), 10))

    sorted_collaboration_tools_votes = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_collaboration_tools_votes = dict(islice(sorted_collaboration_tools_votes.items(), 10))

    sorted_collaboration_tools_answers = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_collaboration_tools_answers = dict(islice(sorted_collaboration_tools_answers.items(), 10))

    sorted_collaboration_tools_comments = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_collaboration_tools_comments = dict(islice(sorted_collaboration_tools_comments.items(), 10))

    sorted_collaboration_tools_views = dict(sorted(collaboration_tools.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_collaboration_tools_views = dict(islice(sorted_collaboration_tools_views.items(), 10))

    answered_collaboration_tools = {k: v for k, v in collaboration_tools.items() if v[1] > 0}

    sorted_collaboration_tools_response_time = dict(sorted(answered_collaboration_tools.items(), key=lambda item: item[1][4]))
    top_10_collaboration_tools_response_time = dict(islice(sorted_collaboration_tools_response_time.items(), 10))

    sorted_collaboration_tools_response_time_reverse = dict(sorted(answered_collaboration_tools.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_collaboration_tools_response_time_reverse = dict(islice(sorted_collaboration_tools_response_time_reverse.items(), 10))

    ##############################################

    sorted_dev_tools_votes = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_dev_tools_votes = dict(islice(sorted_dev_tools_votes.items(), 10))

    sorted_dev_tools_answers = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][1]))
    top_10_dev_tools_answers = dict(islice(sorted_dev_tools_answers.items(), 10))

    sorted_dev_tools_comments = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][2]))
    top_10_dev_tools_comments = dict(islice(sorted_dev_tools_comments.items(), 10))

    sorted_dev_tools_views = dict(sorted(dev_tools.items(), reverse=True, key=lambda item: item[1][3]))
    top_10_dev_tools_views = dict(islice(sorted_dev_tools_views.items(), 10))

    answered_dev_tools = {k: v for k, v in dev_tools.items() if v[1] > 0}

    sorted_dev_tools_response_time = dict(sorted(answered_dev_tools.items(), key=lambda item: item[1][4]))
    top_10_dev_tools_response_time = dict(islice(sorted_dev_tools_response_time.items(), 10))

    sorted_dev_tools_response_time_reverse = dict(sorted(answered_dev_tools.items(), reverse=True, key=lambda item: item[1][4]))
    top_10_dev_tools_response_time_reverse = dict(islice(sorted_dev_tools_response_time_reverse.items(), 10))

    distinct_users = Counter(usernames)
    sorted_distinct_users = dict(sorted(distinct_users.items(), reverse=True, key=lambda item: item[1]))
    top_10_distinct_users = dict(islice(sorted_distinct_users.items(), 10))

    sorted_ids_and_votes = dict(sorted(ids_and_votes.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_votes = dict(islice(sorted_ids_and_votes.items(), 10))

    sorted_ids_and_answers = dict(sorted(ids_and_answers.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_answers = dict(islice(sorted_ids_and_answers.items(), 10))

    sorted_ids_and_comments = dict(sorted(ids_and_comments.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_comments = dict(islice(sorted_ids_and_comments.items(), 10))

    sorted_ids_and_views = dict(sorted(ids_and_views.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_views = dict(islice(sorted_ids_and_views.items(), 10))

    answered_ids_and_response_times = {k: v for k, v in merged_dict.items() if v[2] > 0}

    sorted_ids_and_response_time = dict(sorted(answered_ids_and_response_times.items(), key=lambda item: item[1][0]))
    top_10_sorted_ids_and_response_time = dict(islice(sorted_ids_and_response_time.items(), 10))

    sorted_ids_and_response_time_reverse = dict(sorted(answered_ids_and_response_times.items(), reverse=True, key=lambda item: item[1][0]))
    top_10_sorted_ids_and_response_time_reverse = dict(islice(sorted_ids_and_response_time_reverse.items(), 10))


    numberOfComments = sum(comments)
    avgNumberOfComments = format((numberOfComments / question_number), '.3f')
    numberOfAnswers = sum(answers)
    avgNumberOfAnswers = format((numberOfAnswers / question_number), '.3f')
    numberOfVotes = sum(votes)
    avgNumberOfVotes = format((numberOfVotes / question_number), '.3f')
    yesCounter = 0
    noCounter = 0
    for snippet in code_snippets:
        if snippet == 1:
            yesCounter += 1
        else:
            noCounter += 1

    snippetData = [yesCounter, noCounter]

    sorted_dates = sorted(dates, key=lambda d: tuple(map(int, d.split('-'))))

    for i in range(len(sorted_dates)):
        dates_and_values[sorted_dates[i]] = sorted_dates.count(sorted_dates[i])  # dict for the lineChart

    for i in range(len(tags)):
        tags_and_values[tags[i]] = tags.count(tags[i])  # dict for wordCloud

    sorted_tags_and_values = dict(
        sorted(tags_and_values.items(), reverse=True, key=lambda item: item[1]))  # sorted dict for wordCloud

    best_sorted_tags_and_values = dict(islice(sorted_tags_and_values.items(), 80))  # top 80 for wordCloud

    top_ten_tags_and_values_barchart = dict(islice(sorted_tags_and_values.items(), 10))  # top 10 for barChart
    top_twenty_tags_and_values_chord = dict(islice(sorted_tags_and_values.items(), 20))  # top 20 for chordDiagram
    top_twenty_tags = list(top_twenty_tags_and_values_chord.keys())  # to 10 tag names for the chord diagram

    for key, value in best_sorted_tags_and_values.items():  # map the dict for the wordCloud
        d = {"text": key, "size": value}
        list_of_tags_and_values.append(d)

    labels = list(dates_and_values.keys())  # lineChart labels
    values = list(dates_and_values.values())  # lineChart values

    counter = 0
    days = 0
    previous_value = 0
    halfMonthValues = []
    for key, value in dates_and_values.items():
        counter += value
        if days == 7:
            difference = abs((counter / 7) - previous_value)
            for i in range(7):
                dummy = counter / 7
                if previous_value < dummy:
                    previous_value = previous_value + (difference / 7)
                    halfMonthValues.append(previous_value)
                else:
                    previous_value = previous_value - (difference / 7)
                    halfMonthValues.append(previous_value)
            counter = 0
            days = 0
        days += 1

    added_values = values.copy()

    for i in range(1, len(added_values)):
        added_values[i] = added_values[i] + added_values[i - 1]

    barChartLabels = list(top_ten_tags_and_values_barchart.keys())  # barChart labels
    barChartValues = list(top_ten_tags_and_values_barchart.values())  # barChart values

    list_of_tuples_for_coordinates = [tuple(elem) for elem in coordinates]

    coordinates_counter_dict = dict(Counter(list_of_tuples_for_coordinates))

    coordinates_latitude = []
    coordinates_longitude = []
    coordinates_values = []

    for key, value in coordinates_counter_dict.items():
        coordinates_latitude.append(key[0])
        coordinates_longitude.append(key[1])
        coordinates_values.append(value)

    normalized_coordinates_values = [float(i) / max(coordinates_values) for i in coordinates_values]

    latLngInt = []

    for i in range(len(coordinates_values)):
        latLngInt.append([coordinates_latitude[i], coordinates_longitude[i], normalized_coordinates_values[i]])

    distinct_tags = []
    radar_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]

    stacked_open_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]
    stacked_closed_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]
    stacked_deleted_values = [0, 0, 0, 0, 0, 0, 0]  # [languages,frameworks,big data,dbs,platforms,collab tools,dev tools]

    languages_tags_and_values = {}
    frameworks_tags_and_values = {}
    big_data_ml_tags_and_values = {}
    databases_tags_and_values = {}
    platforms_tags_and_values = {}
    collaboration_tools_tags_and_values = {}
    developer_tools_tags_and_values = {}

    for tag in tags:
        if tag in fields_and_techs.keys():
            if tag not in distinct_tags:
                distinct_tags.append(tag)

    for tag in distinct_tags:
        if fields_and_techs.get(tag) == 'Languages':
            radar_values[0] = radar_values[0] + 1
            languages_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Web Frameworks':
            radar_values[1] = radar_values[1] + 1
            frameworks_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Big Data - ML':
            radar_values[2] = radar_values[2] + 1
            big_data_ml_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Databases':
            radar_values[3] = radar_values[3] + 1
            databases_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Platforms':
            radar_values[4] = radar_values[4] + 1
            platforms_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Collaboration Tools':
            radar_values[5] = radar_values[5] + 1
            collaboration_tools_tags_and_values[tag] = tags.count(tag)
        elif fields_and_techs.get(tag) == 'Developer Tools':
            radar_values[6] = radar_values[6] + 1
            developer_tools_tags_and_values[tag] = tags.count(tag)


    sorted_languages_tags_and_values = dict(
        sorted(languages_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_frameworks_tags_and_values = dict(
        sorted(frameworks_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_big_data_ml_tags_and_values = dict(
        sorted(big_data_ml_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_databases_tags_and_values = dict(
        sorted(databases_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_platforms_tags_and_values = dict(
        sorted(platforms_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_collaboration_tools_tags_and_values = dict(
        sorted(collaboration_tools_tags_and_values.items(), reverse=True, key=lambda item: item[1]))
    sorted_developer_tools_tags_and_values = dict(
        sorted(developer_tools_tags_and_values.items(), reverse=True, key=lambda item: item[1]))  

    top_10_sorted_languages_tags_and_values = dict(islice(sorted_languages_tags_and_values.items(), 10))
    top_10_sorted_frameworks_tags_and_values = dict(islice(sorted_frameworks_tags_and_values.items(), 10))
    top_10_sorted_big_data_ml_tags_and_values = dict(islice(sorted_big_data_ml_tags_and_values.items(), 10))
    top_10_sorted_databases_tags_and_values = dict(islice(sorted_databases_tags_and_values.items(), 10))
    top_10_sorted_platforms_tags_and_values = dict(islice(sorted_platforms_tags_and_values.items(), 10))
    top_10_sorted_collaboration_tools_tags_and_values = dict(islice(sorted_collaboration_tools_tags_and_values.items(), 10))
    top_10_sorted_developer_tools_tags_and_values = dict(islice(sorted_developer_tools_tags_and_values.items(), 10))

    names_top_10_sorted_languages_tags = list(top_10_sorted_languages_tags_and_values.keys())
    names_top_10_sorted_frameworks_tags = list(top_10_sorted_frameworks_tags_and_values.keys())
    names_top_10_sorted_big_data_ml_tags = list(top_10_sorted_big_data_ml_tags_and_values.keys())
    names_top_10_sorted_databases_tags = list(top_10_sorted_databases_tags_and_values.keys())
    names_top_10_sorted_platforms_tags = list(top_10_sorted_platforms_tags_and_values.keys())
    names_top_10_sorted_collaboration_tools_tags = list(top_10_sorted_collaboration_tools_tags_and_values.keys())
    names_top_10_sorted_developer_tools_tags = list(top_10_sorted_developer_tools_tags_and_values.keys())
    

    for i in range(len(radar_values)):
        radar_values[i] = radar_values[i] / len(distinct_tags)

    # Creation of the polar chart open values.
    for question in questions:
        for tag in question['tag'].split(' '):
            if tag in fields_and_techs.keys():
                if fields_and_techs.get(tag) == 'Languages':
                    stacked_open_values[0] = stacked_open_values[0] + 1
                elif fields_and_techs.get(tag) == 'Web Frameworks':
                    stacked_open_values[1] = stacked_open_values[1] + 1
                elif fields_and_techs.get(tag) == 'Big Data - ML':
                    stacked_open_values[2] = stacked_open_values[2] + 1
                elif fields_and_techs.get(tag) == 'Databases':
                    stacked_open_values[3] = stacked_open_values[3] + 1
                elif fields_and_techs.get(tag) == 'Platforms':
                    stacked_open_values[4] = stacked_open_values[4] + 1
                elif fields_and_techs.get(tag) == 'Collaboration Tools':
                    stacked_open_values[5] = stacked_open_values[5] + 1
                elif fields_and_techs.get(tag) == 'Developer Tools':
                    stacked_open_values[6] = stacked_open_values[6] + 1
                    
    # Creation of the polar chart closed values.
    for question in closed_questions:
        for tag in question['tag'].split(' '):
            if tag in fields_and_techs.keys():
                if fields_and_techs.get(tag) == 'Languages':
                    stacked_closed_values[0] += 1
                elif fields_and_techs.get(tag) == 'Web Frameworks':
                    stacked_closed_values[1] += 1
                elif fields_and_techs.get(tag) == 'Big Data - ML':
                    stacked_closed_values[2] += 1
                elif fields_and_techs.get(tag) == 'Databases':
                    stacked_closed_values[3] += 1
                elif fields_and_techs.get(tag) == 'Platforms':
                    stacked_closed_values[4] += 1
                elif fields_and_techs.get(tag) == 'Collaboration Tools':
                    stacked_closed_values[5] += 1
                elif fields_and_techs.get(tag) == 'Developer Tools':
                    stacked_closed_values[6] += 1
    
    # Creation of the polar chart deleted values.
    for question in deleted_questions:
        for tag in question['tag'].split(' '):
            if tag in fields_and_techs.keys():
                if fields_and_techs.get(tag) == 'Languages':
                    stacked_deleted_values[0] += 1
                elif fields_and_techs.get(tag) == 'Web Frameworks':
                    stacked_deleted_values[1] += 1
                elif fields_and_techs.get(tag) == 'Big Data - ML':
                    stacked_deleted_values[2] += 1
                elif fields_and_techs.get(tag) == 'Databases':
                    stacked_deleted_values[3] += 1
                elif fields_and_techs.get(tag) == 'Platforms':
                    stacked_deleted_values[4] += 1
                elif fields_and_techs.get(tag) == 'Collaboration Tools':
                    stacked_deleted_values[5] += 1
                elif fields_and_techs.get(tag) == 'Developer Tools':
                    stacked_deleted_values[6] += 1
    
    languges_question_types = [stacked_open_values[0], stacked_deleted_values[0], stacked_closed_values[0]]
    web_frameworks_question_types = [stacked_open_values[1], stacked_deleted_values[1], stacked_closed_values[1]]
    big_data_ml_question_types = [stacked_open_values[2], stacked_deleted_values[2], stacked_closed_values[2]]
    databases_question_types = [stacked_open_values[3], stacked_deleted_values[3], stacked_closed_values[3]]
    platforms_question_types = [stacked_open_values[4], stacked_deleted_values[4], stacked_closed_values[4]]
    collaboration_tools_question_types = [stacked_open_values[5], stacked_deleted_values[5], stacked_closed_values[5]]
    dev_tools_question_types = [stacked_open_values[6], stacked_deleted_values[6], stacked_closed_values[6]]
    all_groups_question_types = [len(all_filtered_questions_list), len(deleted_questions), len(closed_questions)]

    # Distinct technologies
    distinct_technologies = []
    for tech in fields_and_techs.values():
        if tech not in distinct_technologies:
            distinct_technologies.append(tech)

    # creation of chord diagram matrix
    tag_link_matrix = np.zeros((20, 20)).astype(int)
    tags_to_be_linked = []
    languages_tag_link_matrix = np.zeros((10, 10)).astype(int)
    languages_tags_to_be_linked = []
    frameworks_tag_link_matrix = np.zeros((10, 10)).astype(int)
    frameworks_tags_to_be_linked = []
    big_data_ml_tag_link_matrix = np.zeros((10, 10)).astype(int)
    big_data_ml_tags_to_be_linked = []
    databases_tag_link_matrix = np.zeros((10, 10)).astype(int)
    databases_tags_to_be_linked = []
    platforms_tag_link_matrix = np.zeros((10, 10)).astype(int)
    platforms_tags_to_be_linked = []
    collaborations_tools_tag_link_matrix = np.zeros((10, 10)).astype(int)
    collaborations_tools_tags_to_be_linked = []
    developer_tools_tag_link_matrix = np.zeros((10, 10)).astype(int)
    developer_tools_tags_to_be_linked = []


    for question in questions:
        record_tags = question['tag'].split()
        if [i for i in top_twenty_tags if i in record_tags]:
            for tag in record_tags:
                if tag in top_twenty_tags:
                    tags_to_be_linked.append(top_twenty_tags.index(tag))
            if len(tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        tag_link_matrix[combination[0], combination[1]] += 1
                        tag_link_matrix[combination[1], combination[0]] += 1
            tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_languages_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_languages_tags:
                    languages_tags_to_be_linked.append(names_top_10_sorted_languages_tags.index(tag))
            if len(languages_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(languages_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        languages_tag_link_matrix[combination[0], combination[1]] += 1
                        languages_tag_link_matrix[combination[1], combination[0]] += 1
            languages_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_frameworks_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_frameworks_tags:
                    frameworks_tags_to_be_linked.append(names_top_10_sorted_frameworks_tags.index(tag))
            if len(frameworks_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(frameworks_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        frameworks_tag_link_matrix[combination[0], combination[1]] += 1
                        frameworks_tag_link_matrix[combination[1], combination[0]] += 1
            frameworks_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_big_data_ml_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_big_data_ml_tags:
                    big_data_ml_tags_to_be_linked.append(names_top_10_sorted_big_data_ml_tags.index(tag))
            if len(big_data_ml_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(big_data_ml_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        big_data_ml_tag_link_matrix[combination[0], combination[1]] += 1
                        big_data_ml_tag_link_matrix[combination[1], combination[0]] += 1
            big_data_ml_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_databases_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_databases_tags:
                    databases_tags_to_be_linked.append(names_top_10_sorted_databases_tags.index(tag))
            if len(databases_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(databases_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        databases_tag_link_matrix[combination[0], combination[1]] += 1
                        databases_tag_link_matrix[combination[1], combination[0]] += 1
            databases_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_platforms_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_platforms_tags:
                    platforms_tags_to_be_linked.append(names_top_10_sorted_platforms_tags.index(tag))
            if len(platforms_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(platforms_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        platforms_tag_link_matrix[combination[0], combination[1]] += 1
                        platforms_tag_link_matrix[combination[1], combination[0]] += 1
            platforms_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_collaboration_tools_tags if i in record_tags]:
            for tag in record_tags:
                if tag in names_top_10_sorted_collaboration_tools_tags:
                    collaborations_tools_tags_to_be_linked.append(names_top_10_sorted_collaboration_tools_tags.index(tag))
            if len(collaborations_tools_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(collaborations_tools_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        collaborations_tools_tag_link_matrix[combination[0], combination[1]] += 1
                        collaborations_tools_tag_link_matrix[combination[1], combination[0]] += 1
            collaborations_tools_tags_to_be_linked.clear()
        if [i for i in names_top_10_sorted_developer_tools_tags if i in record_tags]:    
            for tag in record_tags:
                if tag in names_top_10_sorted_developer_tools_tags:
                    developer_tools_tags_to_be_linked.append(names_top_10_sorted_developer_tools_tags.index(tag))
            if len(developer_tools_tags_to_be_linked) > 1:
                combinations_of_tags = list(combinations(developer_tools_tags_to_be_linked, 2))
                for combination in combinations_of_tags:
                    if combination[0] != combination[1]:
                        developer_tools_tag_link_matrix[combination[0], combination[1]] += 1
                        developer_tools_tag_link_matrix[combination[1], combination[0]] += 1
            developer_tools_tags_to_be_linked.clear()

    list_tag_link_matrix = np.array2string(tag_link_matrix, separator=",")
    list_languages_tag_link_matrix = np.array2string(languages_tag_link_matrix, separator=",")
    list_frameworks_tag_link_matrix = np.array2string(frameworks_tag_link_matrix, separator=",")
    list_big_data_ml_tag_link_matrix = np.array2string(big_data_ml_tag_link_matrix, separator=",")
    list_databases_tag_link_matrix = np.array2string(databases_tag_link_matrix, separator=",")
    list_platforms_tag_link_matrix = np.array2string(platforms_tag_link_matrix, separator=",")
    list_collaboration_tools_tag_link_matrix = np.array2string(collaborations_tools_tag_link_matrix, separator=",")
    list_developer_tools_tag_link_matrix = np.array2string(developer_tools_tag_link_matrix, separator=",")

    inclusion_index_dict = {}
    for key, value in tag_combo_frequencies.items():
        tags = key.split(" ")
        try:
            tag_1 = int(sorted_tags_and_values[tags[0]])
            tag_2 = int(sorted_tags_and_values[tags[1]])
            tag_combo_inclusion_index = value / min(tag_1, tag_2)
            if tag_combo_inclusion_index <= 1:
                inclusion_index_dict.update({key: tag_combo_inclusion_index})
        except:
            pass

    inclusion_index_dict = dict(sorted(inclusion_index_dict.items(), key=lambda x: x[1], reverse=True))
    top_10_inclusion_index = dict(islice(inclusion_index_dict.items(), 10))
    
    earliest_date = date_from + '00:00:00Z'
    latest_date = date_to + '00:00:00Z'
    search_terms = request.args.get('searchTerms')
    predict_class_new_docs_dict={}
    if search_terms != None:
        search_terms_list = search_terms.split("; ")
        
        predict_class_new_docs_dict = predict_class_new_docs(search_terms_list)
        predict_topic_new_docs_dict = predict_topic_new_docs(search_terms_list)
                
        predict_topic_new_docs_value_list = predict_topic_new_docs_dict['topic'][0]
        i=0
        for key, value_list in predict_class_new_docs_dict.items():
            value_list.append(predict_topic_new_docs_value_list[i])
            i+=1
        
        
    res_pop_dif=topic_popularity_difficulty(earliest_date,latest_date)
    avg_score=res_pop_dif[0]["Avg_score"].tolist()
    avg_views=res_pop_dif[0]["Avg_views"].tolist()
    avg_answers=res_pop_dif[1]["Avg_answers"].tolist()
    avg_hours_to_first_answer=res_pop_dif[1]["Avg_hrs_to_first_answer"].tolist()
    pd=res_pop_dif[1]["PD"].tolist()
    

    res_dif_0_formatted = {}
    for idx, row in res_pop_dif[0].iterrows():
        res_dif_0_formatted[idx] = row.values.tolist()
        
    res_dif_1_formatted = {}
    for idx, row in res_pop_dif[1].iterrows():
        res_dif_1_formatted[idx] = row.values.tolist()
        
    res_growth=growth_topic(earliest_date, latest_date)
    topic_length_early = res_growth['topic_length_early']
    print(topic_length_early)
    topic_length_late = res_growth['topic_length_late']
    share_early = res_growth['share_early']
    share_late = res_growth['share_late']
    self_grown = res_growth['self_grown']
    all_growth = res_growth['all_growth']
    res_growth_dict = {}

    for key in res_growth:
        for i, value in enumerate(res_growth[key]):
            if i not in res_growth_dict:
                res_growth_dict[i] = []
            res_growth_dict[i].append(value)
    
    topic_no_1 = request.args.get('topicNo1')
    topic_no_2 = request.args.get('topicNo2')
    metric_option = request.args.get('metric')
    print(metric_option)
    ptc_dict = {}
    if (topic_no_1 != None and topic_no_2 != None and metric_option != None) :
        if metric_option == 'views':
            metric=documents['views']
        elif metric_option == 'votes':
            metric=documents['votes']
        else:
           metric=documents['answers'] 
           
        
        ptc=pairwise_topic_comparisons(int(topic_no_1),int(topic_no_2),metric,earliest_date,latest_date)
    
        for key in ptc:
            for i, value in enumerate(ptc[key]):
                if i not in ptc_dict:
                    ptc_dict[i] = []
                ptc_dict[i].append(value)


    return render_template('index.html',  questions=questions, question_count=question_count, users=users, labels=labels,
                           values=values,
                           list_of_tags_and_values=list_of_tags_and_values, barChartLabels=barChartLabels,
                           barChartValues=barChartValues, latLngInt=latLngInt, latitudes=latitudes,
                           longitudes=longitudes,
                           distinct_technologies=distinct_technologies,
                           stacked_open_values=stacked_open_values, stacked_closed_values=stacked_closed_values,
                           stacked_deleted_values=stacked_deleted_values,
                           radar_values=radar_values, added_values=added_values, avgNumberOfAnswers=avgNumberOfAnswers,
                           avgNumberOfComments=avgNumberOfComments, avgNumberOfVotes=avgNumberOfVotes,
                           snippetData=snippetData, halfMonthValues=halfMonthValues,
                           top_10_sorted_ids_and_votes=top_10_sorted_ids_and_votes,
                           top_10_sorted_ids_and_answers=top_10_sorted_ids_and_answers,
                           top_10_sorted_ids_and_comments=top_10_sorted_ids_and_comments,
                           top_10_sorted_ids_and_views=top_10_sorted_ids_and_views,
                           top_10_distinct_users=top_10_distinct_users, location_name=location_name,
                           location_question=location_question, locations=locations,
                           top_10_languages_votes=top_10_languages_votes,
                           top_10_languages_answers=top_10_languages_answers,
                           top_10_languages_comments=top_10_languages_comments,
                           top_10_languages_views=top_10_languages_views,
                           top_10_web_frameworks_votes=top_10_web_frameworks_votes,
                           top_10_web_frameworks_answers=top_10_web_frameworks_answers,
                           top_10_web_frameworks_comments=top_10_web_frameworks_comments,
                           top_10_web_frameworks_views=top_10_web_frameworks_views,
                           top_10_big_data_ml_votes=top_10_big_data_ml_votes,
                           top_10_big_data_ml_answers=top_10_big_data_ml_answers,
                           top_10_big_data_ml_comments=top_10_big_data_ml_comments,
                           top_10_big_data_ml_views=top_10_big_data_ml_views,
                           top_10_databases_votes=top_10_databases_votes,
                           top_10_databases_answers=top_10_databases_answers,
                           top_10_databases_comments=top_10_databases_comments,
                           top_10_databases_views=top_10_databases_views,
                           top_10_platforms_votes=top_10_platforms_votes,
                           top_10_platforms_answers=top_10_platforms_answers,
                           top_10_platforms_comments=top_10_platforms_comments,
                           top_10_platforms_views=top_10_platforms_views,
                           top_10_collaboration_tools_votes=top_10_collaboration_tools_votes,
                           top_10_collaboration_tools_answers=top_10_collaboration_tools_answers,
                           top_10_collaboration_tools_comments=top_10_collaboration_tools_comments,
                           top_10_collaboration_tools_views=top_10_collaboration_tools_views,
                           top_10_dev_tools_votes=top_10_dev_tools_votes,
                           top_10_dev_tools_answers=top_10_dev_tools_answers,
                           top_10_dev_tools_comments=top_10_dev_tools_comments,
                           top_10_dev_tools_views=top_10_dev_tools_views,
                           date_from=date_from, date_to=date_to, 
                           list_tag_link_matrix=list_tag_link_matrix, top_twenty_tags=top_twenty_tags, 
                           list_languages_tag_link_matrix = list_languages_tag_link_matrix, names_top_10_sorted_languages_tags = names_top_10_sorted_languages_tags,
                           list_frameworks_tag_link_matrix = list_frameworks_tag_link_matrix, names_top_10_sorted_frameworks_tags = names_top_10_sorted_frameworks_tags,
                           list_big_data_ml_tag_link_matrix = list_big_data_ml_tag_link_matrix, names_top_10_sorted_big_data_ml_tags = names_top_10_sorted_big_data_ml_tags,
                           list_databases_tag_link_matrix = list_databases_tag_link_matrix, names_top_10_sorted_databases_tags = names_top_10_sorted_databases_tags,
                           list_platforms_tag_link_matrix = list_platforms_tag_link_matrix, names_top_10_sorted_platforms_tags = names_top_10_sorted_platforms_tags,
                           list_collaboration_tools_tag_link_matrix = list_collaboration_tools_tag_link_matrix, names_top_10_sorted_collaboration_tools_tags = names_top_10_sorted_collaboration_tools_tags,
                           list_developer_tools_tag_link_matrix = list_developer_tools_tag_link_matrix, names_top_10_sorted_developer_tools_tags = names_top_10_sorted_developer_tools_tags,
                           times_data_list = times_data_list, survival_time_curve_values = survival_time_curve_values, 
                           languages_times_data_list = languages_times_data_list, languages_survival_time_curve_values = languages_survival_time_curve_values,
                           web_frameworks_times_data_list = web_frameworks_times_data_list, web_frameworks_survival_time_curve_values = web_frameworks_survival_time_curve_values,
                           big_data_ml_times_data_list = big_data_ml_times_data_list, big_data_ml_survival_time_curve_values = big_data_ml_survival_time_curve_values,
                           databases_times_data_list = databases_times_data_list, databases_survival_time_curve_values = databases_survival_time_curve_values,
                           platforms_times_data_list = platforms_times_data_list, platforms_survival_time_curve_values = platforms_survival_time_curve_values,
                           collaboration_tools_times_data_list = collaboration_tools_times_data_list, collaboration_tools_survival_time_curve_values = collaboration_tools_survival_time_curve_values,
                           dev_tools_times_data_list = dev_tools_times_data_list, dev_tools_survival_time_curve_values = dev_tools_survival_time_curve_values,
                           languges_question_types = languges_question_types,
                            web_frameworks_question_types = web_frameworks_question_types,
                            big_data_ml_question_types = big_data_ml_question_types,
                            databases_question_types = databases_question_types,
                            platforms_question_types = platforms_question_types,
                            collaboration_tools_question_types = collaboration_tools_question_types,
                            dev_tools_question_types = dev_tools_question_types,
                            all_groups_question_types = all_groups_question_types,
                            sorted_comments_distribution_labels = sorted_comments_distribution_labels,
                            sorted_comments_distribution_values = sorted_comments_distribution_values,
                            sorted_answers_distribution_labels = sorted_answers_distribution_labels,
                            sorted_answers_distribution_values = sorted_answers_distribution_values,
                            sorted_votes_distribution_labels = sorted_votes_distribution_labels,
                            sorted_votes_distribution_values = sorted_votes_distribution_values,
                            sorted_views_distribution_labels = sorted_views_distribution_labels,
                            sorted_views_distribution_values = sorted_views_distribution_values,
                            answeredData = answeredData, top_10_inclusion_index = top_10_inclusion_index,
                            median_times_dict = median_times_dict, 
                            top_10_languages_response_time = top_10_languages_response_time,
                            top_10_web_frameworks_response_time = top_10_web_frameworks_response_time,
                            top_10_big_data_ml_response_time = top_10_big_data_ml_response_time,
                            top_10_databases_response_time = top_10_databases_response_time,
                            top_10_platforms_response_time = top_10_platforms_response_time,
                            top_10_collaboration_tools_response_time = top_10_collaboration_tools_response_time,
                            top_10_dev_tools_response_time = top_10_dev_tools_response_time,
                            top_10_sorted_ids_and_response_time = top_10_sorted_ids_and_response_time,
                            top_10_languages_response_time_reverse=top_10_languages_response_time_reverse,
                            top_10_web_frameworks_response_time_reverse=top_10_web_frameworks_response_time_reverse,
                            top_10_big_data_ml_response_time_reverse=top_10_big_data_ml_response_time_reverse,
                            top_10_databases_response_time_reverse=top_10_databases_response_time_reverse,
                            top_10_platforms_response_time_reverse=top_10_platforms_response_time_reverse,
                            top_10_collaboration_tools_response_time_reverse=top_10_collaboration_tools_response_time_reverse,
                            top_10_dev_tools_response_time_reverse=top_10_dev_tools_response_time_reverse,
                            top_10_sorted_ids_and_response_time_reverse=top_10_sorted_ids_and_response_time_reverse,
                            predict_class_new_docs_dict = predict_class_new_docs_dict,
                            res_dif_0_formatted = res_dif_0_formatted, res_dif_1_formatted = res_dif_1_formatted,
                            res_growth_dict = res_growth_dict, share_late = share_late, all_growth = all_growth,
                            avg_score = avg_score, pd = pd, avg_views = avg_views, avg_answers = avg_answers, 
                            avg_hours_to_first_answer = avg_hours_to_first_answer, topic_length_early = topic_length_early,
                            topic_length_late = topic_length_late, share_early = share_early, self_grown = self_grown, ptc_dict = ptc_dict
                           )

if __name__ == "__main__":
    app.run(debug=True)