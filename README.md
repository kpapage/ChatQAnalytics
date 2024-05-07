# ChatQAnalytics Dashboard
## Instalation Instructions
1. Download and Install Mongo DB

2. Create a database named LLM-db

3. Import to the database the **_questions_** table

4. Install the Python Flask Micro Web Framework by running the command :
```
pip install flask
```
5. Install the bertopic Python library and fix subpackage versions
```
pip install bertopic

pip uninstall numba

pip install numba==0.57.0
```  

7. Clone the repository and open the project in your IDE

8. Run the command to start the localhost server on port 5000:
```
flask run
```
8. Visit the http://localhost:5000/ url on your browser and the **_ChatQAnalytics Dashboard_**  will be located there



## About the Project
ChatQAnalytics is a collective dashboard of SO data that cleverly presents and visualizes the technological trends extracted from SO data related with ChatGPT. More specifically, ChatQAnalytics comprises an efficient platform that aims to:
1. Accumulate data of multiple forms (geographical coordinates, unstructured text, numerical)
2. Store and analyze the collected data
3. Visualize the data using appropriate tools and packages
4. Present the data in an efficient manner

