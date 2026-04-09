# AI/ML Based College Chatbot - SRI VENKATESHWARA POLYTECHNIC

## Project Highlights
- NLP preprocessing: tokenization, stopword removal, stemming, TF-IDF
- ML intent classification with Logistic Regression (or Naive Bayes)
- MySQL knowledge base with admin dashboard
- Real chat UI + optional voice input

## Folder Structure
- `app.py` Flask backend
- `train.py` ML training script
- `seed_db.py` load CSV into MySQL
- `data/intents.csv` sample dataset
- `models/` trained artifacts
- `templates/` HTML pages
- `static/` CSS/JS assets
- `schema.sql` MySQL schema

## AI/ML Software Development Life Cycle
1. Problem Definition: Answer college FAQs using AI/ML
2. Data Collection: Q&A dataset in `data/intents.csv`
3. Data Preprocessing: cleaning, tokenization, stopwords, stemming
4. Feature Extraction: TF-IDF vectorization
5. Model Training: Logistic Regression / Naive Bayes
6. Evaluation: accuracy on train/test split
7. Deployment: Flask API + web UI
8. Monitoring & Updates: add new Q&A via Admin, retrain model

## How to Run (VS Code)
1. Open this folder in VS Code.
2. Create a virtual environment and install requirements.
3. Create the MySQL database using `schema.sql`.
4. Seed the database and train the model.
5. Run the Flask app.

See instructions in chat response for exact commands.
