# ChatQAnalytics Dashboard

## Installation Instructions

### 1. Set Up MongoDB
- Download and install **MongoDB** from the official website: [MongoDB Installation](https://www.mongodb.com/try/download/community).
- After installation, create a database named `LLM-db`.

### 2. Set Up the Collection
- Create a collection named **_questions_** in your `LLM-db` database.
- Import the **_LLMsite/static/models/LLM-db.questions.csv_** file into the **_questions_** collection.

### 3. Clone the Project
- Open the cloned project in your preferred Integrated Development Environment (IDE).

### 4. Install Dependencies
- Install the required dependencies for the ChatQAnalytics dashboard by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```

### 5. Start the Localhost Server
- To start the local server, run:
    ```bash
    flask run
    ```
- The ChatQAnalytics Dashboard will be available at `http://localhost:5000/` in your browser.

---

## Scraper Instructions - Used when the user wants to reproduce the experiments with different data

### 1. Install Dependencies for the Scraper
- Navigate to the **_scrapers_** directory and run the following command to install all necessary dependencies:
    ```bash
    pip install -r scraper_requirements.txt
    ```

### 2. Set Up the `.env` File
- Inside the **_scrapers_** directory, create a `.env` file and add your API key as follows:
    ```bash
    API_KEY='your_api_key'
    ```

### 3. Modify the Search String
- Modify the **_search_string_** variable on line 164 of `LLMscraper.py` to your desired search query.

### 4. Run the Scraper
- Navigate to the **_scrapers_** directory and run the scraper with the following command:
    ```bash
    python LLMscraper.py
    ```

### 5. Filter and Export Data
- After running the scraper, filter the collected data in your MongoDB database as needed.
- Export the filtered data into a CSV file named **_LLM-db.questions_**.

### 6. Place the Exported CSV in the Appropriate Directory
- Move the exported CSV file to the **_LLMsite/static/models_** directory.

### 7. Run the Preprocessing Script. 
- This script trains a BERTopic model, Tag clustering visualizations and a Random Forest model using the titles of the questions. The models and visualizations are stored in the appropriate paths within the project. For the Random Forest model, we currently maintain a simple approach for training a Random Forest, while the finetuning parameters used in the experiements of the current work can be found in comments (lines 267-276).
- For the models, File 'LLMsite/static/models/LLM-db.questions.csv' stores the positive samples and file 'LLMsite/static/models/Newest_records_SO.csv' stores the negative samples. The latter file may change as well to store different questions than the ones selected from the authors. To not alter the script, we advise that this file contains the title of the questions in a column described as 'Title'.
- Simple approach to run the script: Navigate to the root directory of the repository and execute the following command to process the data:
    ```bash
    python preprocessing_phase_steps.py
    ```
- Within the script, users may modify the parameters of the models to adapt to their needs, e.g. more/less topics, adjust finetuning parameters in Random Forest. In any case, the user can review the script to train one model at a time instead of replacing all models at once.
- This script can be used at any time to alter the models and review the dashboard that functions based on the properties of these models.

### 8. View the Dashboard
- With the data now processed, you can now start the Localhost again, and visit the ChatQAnalytics dashboard at `http://localhost:5000/` to browse the updated data for your search string.

---

## About the Project

ChatQAnalytics is an advanced dashboard for analyzing Stack Overflow (SO) data, focusing on the technological trends related to **ChatGPT**. The platform provides an efficient and user-friendly interface to:

1. **Accumulate** various types of data, including geographical coordinates, unstructured text, and numerical information.
2. **Store** and **analyze** the collected data in a MongoDB database.
3. **Visualize** the data through powerful tools and visualization techniques.
4. **Present** the analyzed data in an intuitive and meaningful way, making it easy to identify trends and insights.
