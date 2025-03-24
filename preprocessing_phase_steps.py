
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

#Load the data from the default path of the project.
#Alternatively, you can load other data with identical format and develop new models. In this case, you must update the mongo database as well indicated in the readme file of the project.
documents = pd.read_csv("LLMsite/static/models/LLM-db.questions.csv")
documents["question_title"] = documents["question_title"].astype(str)

#Load the SentenceTransformer used for producing contextualized embeddings, you can select alternative options.
#The embeddings are used in BERTopic and Text classification, i.e. training Random Forest model.
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

#This function extracts the post titles from the file containing the Stack Overflow questions (documents - see above)
#After, a BERTopic model is is trained using the default options. Addtional configurations can be found in https://maartengr.github.io/BERTopic/index.html#common
#This function stores the BERTopic model in the right path of the project.
#Also, the funcitons stores HTML files representing the main outputs of BERTopic in the right paths of the project: 1) the hierarchy of topics, 2) a 2d topic visualization and, 3) a 2d document visualization.
def store_bertopic(documents):

    docs = list(documents.loc[:, "question_title"].values)

    embeddings = sentence_model.encode(docs, show_progress_bar=True)

    umap_model = UMAP(random_state=123)

    #Train BERTopic model
    #You can adjust the number of topics using the parameter nr_topics
    model = BERTopic(verbose=True, umap_model=umap_model)


    model.fit(docs,embeddings)
    vis_topics = model.visualize_topics()
    #vis_topics.show()

    model.nr_topics=len(model.topic_embeddings_)
    model.save("LLMsite/static/models/Bertopic_model_reduced_30_topics")

    #raw_html = vis_topics._repr_html_()
    raw_html=vis_topics.to_html()

    with open("LLMsite/templates/topic_visualization/bert_visualize_reduced.html", "w", encoding="utf-8") as file:
        file.write(raw_html)


    reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    vis_documents=model.visualize_documents(docs=docs,embeddings=embeddings)
    raw_html=vis_documents.to_html()

    with open("LLMsite/templates/topic_visualization/bert_visualize_docs_reduced.html", "w", encoding="utf-8") as file:
        file.write(raw_html)


    hierarchical_topics = model.hierarchical_topics(docs)
    with open("LLMsite/templates/topic_visualization/bert_visualize_hierarchy_reduced.html", "w",encoding="utf-8") as file:
        file.write(model.visualize_hierarchy(hierarchical_topics=hierarchical_topics,color_threshold=-1).to_html())

store_bertopic(documents)

#This function extracts the tags from the Stack Overflow questions, which are separated with empty space.
#Based on tag co-occurences, the function trains an affinity propagation model and stores a visualization of the clusters in the right path of the project.
#param documents: The data structure storing the questions from Stack Overflow
#param filter_tags: Whether to exclude some tags manually
#param exclude_tags: A list containing the tags to be excluded. In this case we excluded the tags that were used to form the dataset.
#param tag_occur_thres: A threshold for excluding insignificant tags. Only tags with equal or higher frequencies are included in the analysis.
def store_tag_cluster_results(documents,filter_tags=True,tag_occur_thres=3,exclude_tags=["chatgpt-api", "openai-api", "gpt-2", "gpt-3", "gpt-3.5", "gpt-4", "chat-gpt-4","gpt4all"]):
    import numpy as np

    tag_list=documents['tag'].to_list()

    for i in range(len(tag_list)):
        temp=tag_list[i]
        temp=temp.split(" ")
        tag_list[i]=temp


    from itertools import chain
    unique_tags = list(set(list(chain(*tag_list))))

    num_rows = len(documents)
    num_columns = len(unique_tags)

    document_tag_matrix=np.zeros((num_rows, num_columns))


    for i in range(len(tag_list)):
        temp=tag_list[i]
        for j in temp:
            index_pos = unique_tags.index(j)
            document_tag_matrix[i,index_pos]=1


    tag_to_tag_matrix=np.dot(np.transpose(document_tag_matrix) , document_tag_matrix)

    tag_to_tag_matrix_ii=tag_to_tag_matrix

    diag_values_tag=np.diagonal(tag_to_tag_matrix[:,:]).copy()

    for i in range(len(tag_to_tag_matrix)-1):
        for j in range(i+1,len(tag_to_tag_matrix)):
            temp_val=min(diag_values_tag[i],diag_values_tag[j])
            temp_val= tag_to_tag_matrix_ii[i,j]/temp_val
            tag_to_tag_matrix_ii[i,j]=temp_val
            tag_to_tag_matrix_ii[j,i]=temp_val



    np.fill_diagonal(tag_to_tag_matrix_ii,1)

    tag_to_tag_matrix_ii=pd.DataFrame(tag_to_tag_matrix_ii)
    tag_to_tag_matrix_ii.columns=unique_tags




    include_tags_pos=[]

    for i in range(len(diag_values_tag)):
        if diag_values_tag[i]>=tag_occur_thres:
            include_tags_pos.append(i)


    unique_tags=[unique_tags[i] for i in include_tags_pos]
    tag_to_tag_matrix_ii=tag_to_tag_matrix_ii.loc[tag_to_tag_matrix_ii.index[include_tags_pos],tag_to_tag_matrix_ii.columns[include_tags_pos]]


    if filter_tags:
        #exclude_tags=["chatgpt-api", "openai-api", "gpt-2", "gpt-3", "gpt-3.5", "gpt-4", "chat-gpt-4","gpt4all"]
        exclude_tags_pos=[]
        include_tags_pos=[]
        for i in range(len(unique_tags)):
           if unique_tags[i] not in exclude_tags:
               include_tags_pos.append(i)
           else:
               exclude_tags_pos.append(i)

        tag_to_tag_matrix_ii=tag_to_tag_matrix_ii.loc[tag_to_tag_matrix_ii.index[include_tags_pos],tag_to_tag_matrix_ii.columns[include_tags_pos]]
        unique_tags=[unique_tags[i] for i in include_tags_pos]



    from sklearn.cluster import AffinityPropagation
    import numpy as np

    # Create and fit the AffinityPropagation model
    model_ap = AffinityPropagation(affinity='precomputed',random_state=123,verbose=True)#
    model_ap.fit(tag_to_tag_matrix_ii)

    # Get cluster labels
    cluster_labels = model_ap.labels_
    cluster_indices= model_ap.cluster_centers_indices_


    import math
    from pyvis.network import Network

    net = Network(height="750px", width="100%", bgcolor="white",directed=True)#, font_color="white"

    palette = list(np.random.choice(range(256), size=(len(unique_tags),3)))

    node_size=[]

    for i in range(len(unique_tags)):
        color_now=palette[cluster_labels[i]]
        color_now=f"rgb({color_now[0]},{color_now[1]},{color_now[2]})"

        if i in cluster_indices:
            net.add_node(unique_tags[i], label=unique_tags[i],color=color_now,size=500)#,group=cluster_labels[i]
            node_size.append(1000)
        else:
            net.add_node(unique_tags[i], label=unique_tags[i],color=color_now,size=50)#,group=cluster_labels[i]
            node_size.append(100)

        net.nodes[i]['font']={"size":node_size[i],"color":color_now}


    edges = []

    for i in range(len(unique_tags)):
        # print(i)
        if i not in cluster_indices:
            temp_edge = (unique_tags[cluster_indices[cluster_labels[i]]], unique_tags[i])
            edges.append([temp_edge])

    # Add edges

    for edge in edges:
        net.add_edge(edge[0][0], edge[0][1])

        # Set the physics layout of the network
    net.barnes_hut()

    # Show the network
    net.show_buttons(filter_=['physics'])


    net.write_html("LLMsite/templates/topic_visualization/ap_ii_network.html")


    #with open("ap_ii_network.html", "w", encoding="utf-8") as file:
    #   file.write(net.to_json())



    ###Use networkD3

store_tag_cluster_results(documents=documents)

##New documents with negative samples. Used for training classification models
new_docs = pd.read_csv("LLMsite/static/models/Newest_records_SO.csv")

#This function develops a Random Forest classifier for predicting relevant and irrelevant questions, using question titles.
#The final model is selected based on a grid search strategy which evaluates different combinations of parameters (1620 overall) using 10 fold cross-validation.
#For each parameter, the values examined can be altered inline to either simplify the analysis or explore other potential values/combinations.
#When the optimal model is identified, the function stores it in the appropriate path of the project.
def store_random_forest_classifier(new_docs):

    docs = list(documents.loc[:, "question_title"].values)

    embeddings = sentence_model.encode(docs, show_progress_bar=True)


    not_same_pos = []
    for i in range(len(new_docs['Id'])):
        if (new_docs['Id'][i] not in documents['_id']):
            if (new_docs['Title'][i] != ""):
                not_same_pos.append(i)

    new_docs = new_docs.loc[new_docs.index[not_same_pos], :]

    new_docs['Title'] = new_docs.loc[:, "Title"].astype(str)
    docs = list(new_docs.loc[:, "Title"].values)

    new_embs = sentence_model.encode(docs, show_progress_bar=True)

    Y=[1 for i in range(len(embeddings))]
    Y.extend([0 for i in range(len(new_embs))])

    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier

    #Simplified parameters
    """
    param_grid={
        'n_estimators':[3],
        'max_depth':[10],
        'min_samples_split': [10],
        'min_samples_leaf': [4],
        'bootstrap': [True, False],
        'class_weight': ['balanced']
    }
    """

    #Default parameters
    #'''
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    #'''


    reg = RandomForestClassifier(random_state=123)

    from sklearn.metrics import roc_auc_score, make_scorer
    auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)

    grid_search = GridSearchCV(estimator=reg, param_grid=param_grid,
                               cv=10, n_jobs=4, verbose=3, scoring=auc_scorer)  # precision or recall
    # Fit the model to the training data
    grid_search.fit(pd.concat([pd.DataFrame(embeddings), pd.DataFrame(new_embs)], ignore_index=True), Y)

    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best score from GridSearchCV: {grid_search.best_score_}")

    # Save the GridSearchCV object
    import pickle
    #with open('grid_search.pkl', 'wb') as f:
    #    pickle.dump(grid_search, f)
    # Get the best estimator from the grid search
    best_rf = grid_search.best_estimator_
    # Save the model to a file
    joblib.dump(best_rf, 'LLMsite/static/models/random_forest_model.joblib')

store_random_forest_classifier(new_docs)

