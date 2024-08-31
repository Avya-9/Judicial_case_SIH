import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the spacy language model
nlp = spacy.load("en_core_web_sm")
pd.set_option("display.max_columns", None)

cd = pd.read_csv("case_details.csv")
ca = pd.read_csv("case_acts.csv")
cd_ca = pd.merge(cd, ca, on="cino", how="left")
cd_ca.rename(columns={"act": "id"}, inplace=True)
dictionary = dict(zip(ca["cino"], ca["act"]))
act = pd.read_csv("act.csv")
cd_ca_act = pd.merge(cd_ca, act, on="id", how="left")


def clean_text(text):
    if pd.isna(text) or text.lower() == "nil":
        return " "
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


columns_to_combine = ["child_labour", "child_marriage"]
cd_ca_act["CaseType"] = cd_ca_act[columns_to_combine].idxmax(axis=1)
column_mapping = {"child_labour": "Child Labour", "child_marriage": "Child Marriage"}
cd_ca_act["CaseType"] = cd_ca_act["CaseType"].map(column_mapping)

numerical_features = cd_ca_act.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = cd_ca_act.select_dtypes(include=["object", "category"]).columns
catdf = cd_ca_act[categorical_cols]
date_pattern = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}")
date_cols = [
    col
    for col
    in catdf.columns
    if catdf[col].astype(str).str.contains(date_pattern).any()
]
catdf = catdf.drop(date_cols, axis=1)
catdf = catdf.applymap(clean_text)
catdf["Combined"] = catdf.apply(lambda row: " ".join(row.astype(str)), axis=1)

df = cd_ca_act
df = pd.concat([df, catdf["Combined"]], axis=1)


def preprocess_text_spacy(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    processed_text = " ".join(words)
    return processed_text


df["Combined"] = df["Combined"].apply(preprocess_text_spacy)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["Combined"])
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

user_query = "Child Marriage case with pending outcome"
user_query_vector = vectorizer.transform([user_query])

similarities = cosine_similarity(user_query_vector, X)
top_n_indices = similarities[0].argsort()[-2:][::-1]
similar_cases = df.iloc[top_n_indices]
similar_cases = similar_cases.drop(columns=["Combined"])

# Print the output in a better way
print("Similar Cases:")
print(similar_cases.to_string(index=False))
