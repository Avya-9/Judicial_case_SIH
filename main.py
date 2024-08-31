from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the FastAPI app
app = FastAPI()

# Initialize the Spacy language model
nlp = spacy.load("en_core_web_sm")

# Load the CSV files once when the application starts
cd = pd.read_csv("case_details.csv")
ca = pd.read_csv("case_acts.csv")
act = pd.read_csv("act.csv")

# Data processing to combine and clean data
cd_ca = pd.merge(cd, ca, on="cino", how="left")
cd_ca.rename(columns={"act": "id"}, inplace=True)
cd_ca_act = pd.merge(cd_ca, act, on="id", how="left")

# Columns to combine for CaseType
columns_to_combine = ["child_labour", "child_marriage"]
column_mapping = {"child_labour": "Child Labour", "child_marriage": "Child Marriage"}


def clean_text(text):
    if pd.isna(text) or text.lower() == "nil":
        return " "
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Preprocess text using Spacy
def preprocess_text_spacy(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(words)


# Apply transformations
cd_ca_act["CaseType"] = cd_ca_act[columns_to_combine].idxmax(axis=1)
cd_ca_act["CaseType"] = cd_ca_act["CaseType"].map(column_mapping)

categorical_cols = cd_ca_act.select_dtypes(include=["object", "category"]).columns
catdf = cd_ca_act[categorical_cols]
date_pattern = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}")
date_cols = [
    col
    for col in catdf.columns
    if catdf[col].astype(str).str.contains(date_pattern).any()
]
catdf = catdf.drop(date_cols, axis=1)
catdf = catdf.applymap(clean_text)
catdf["Combined"] = catdf.apply(lambda row: " ".join(row.astype(str)), axis=1)
catdf["Combined"] = catdf["Combined"].apply(preprocess_text_spacy)

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(catdf["Combined"])


# API request model
class Query(BaseModel):
    user_query: str


@app.post("/find_similar_cases/")
def find_similar_cases(query: Query):
    try:
        user_query_vector = vectorizer.transform([query.user_query])
        similarities = cosine_similarity(user_query_vector, X)
        top_n_indices = similarities[0].argsort()[-2:][::-1]
        similar_cases = catdf.iloc[top_n_indices]
        if similar_cases.empty:
            return {"message": "No similar cases found"}

        return similar_cases.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
def startup_event():
    print("Server is up and running...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
