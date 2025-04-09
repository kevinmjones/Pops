from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
import pandas as pd
from io import StringIO
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"csv"}

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def preprocess_data(df):
    required_columns = ['Idea reference', 'Idea name', 'Idea status', 'Idea description']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df['Idea name'] = df['Idea name'].fillna("")
    df['Idea description'] = df['Idea description'].fillna("")
    df['combined_text'] = df['Idea name'] + " " + df['Idea description']
    df['Idea status_lower'] = df['Idea status'].str.lower().str.strip()
    return df

def get_recommendations(df, threshold):
    needs_review_df = df[df['Idea status_lower'] == 'needs review'].copy()
    others_df = df[df['Idea status_lower'] != 'needs review'].copy()
    if needs_review_df.empty or others_df.empty:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = pd.concat([needs_review_df['combined_text'], others_df['combined_text']])
    vectorizer.fit(corpus)
    needs_vectors = vectorizer.transform(needs_review_df['combined_text'])
    others_vectors = vectorizer.transform(others_df['combined_text'])
    similarity_matrix = cosine_similarity(needs_vectors, others_vectors)
    recommendations = []
    for idx, similarities in enumerate(similarity_matrix):
        best_match_idx = similarities.argmax()
        best_score = similarities[best_match_idx]
        if best_score > threshold:
            needs_ref = needs_review_df.iloc[idx]['Idea reference']
            candidate_ref = others_df.iloc[best_match_idx]['Idea reference']
            recommendations.append({
                'Needs Review Idea': needs_ref,
                'Recommended Merge Candidate': candidate_ref,
                'Similarity Score': best_score
            })
    return recommendations

# New helper functions for Production Support Triage
def preprocess_support_data(df):
    required_columns = ['Issue key', 'Summary', 'Description', 'status']
    # Create a mapping from lowercase column names to the original column names
    columns_map = {col.lower(): col for col in df.columns}
    for req in required_columns:
        if req.lower() not in columns_map:
            raise ValueError(f"Missing required column: {req}")
    # Rename columns to standard names for required columns
    rename_dict = {columns_map[req.lower()]: req for req in required_columns if req.lower() in columns_map}
    df = df.rename(columns=rename_dict)
    df['Summary'] = df['Summary'].fillna("")
    df['Description'] = df['Description'].fillna("")
    df['combined_text'] = df['Summary'] + " " + df['Description']
    df['ticket_status_lower'] = df['status'].str.lower().str.strip()
    return df

def get_support_recommendations(df, threshold):
    # Tickets with status "open" are the new tickets that need triage.
    open_df = df[df['ticket_status_lower'] == 'open'].copy()
    others_df = df[df['ticket_status_lower'] != 'open'].copy()
    if open_df.empty or others_df.empty:
        return []
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = pd.concat([open_df['combined_text'], others_df['combined_text']])
    vectorizer.fit(corpus)
    open_vectors = vectorizer.transform(open_df['combined_text'])
    others_vectors = vectorizer.transform(others_df['combined_text'])
    similarity_matrix = cosine_similarity(open_vectors, others_vectors)
    recommendations = []
    for idx, similarities in enumerate(similarity_matrix):
        best_match_idx = similarities.argmax()
        best_score = similarities[best_match_idx]
        if best_score > threshold:
            ticket_id = open_df.iloc[idx]['Issue key']
            candidate_id = others_df.iloc[best_match_idx]['Issue key']
            recommendations.append({
                'Ticket ID': ticket_id,
                'Recommended Ticket': candidate_id,
                'Similarity Score': best_score
            })
    return recommendations

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/new_ideas_merge_review", methods=["GET", "POST"])
def new_ideas_merge_review():
    recommendations = []
    if request.method == "POST":
        if "ideas_file" not in request.files:
            return "No file part", 400
        file = request.files["ideas_file"]
        if file.filename == "" or not allowed_file(file.filename):
            return "Invalid file", 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except Exception as e:
            return f"Error processing CSV: {e}", 500
        try:
            df = preprocess_data(df)
        except Exception as e:
            return f"Preprocessing error: {e}", 500
        try:
            threshold = float(request.form.get("threshold", "0.85"))
        except ValueError:
            threshold = 0.85
        recommendations = get_recommendations(df, threshold)
        recommendations = sorted(recommendations, key=lambda x: x['Similarity Score'], reverse=True)
    return render_template("new_ideas_merge_review.html", recommendations=recommendations)

# New route for Production Support Triage
@app.route("/production_support_triage", methods=["GET", "POST"])
def production_support_triage():
    recommendations = []
    if request.method == "POST":
        if "tickets_file" not in request.files:
            return "No file part", 400
        file = request.files["tickets_file"]
        if file.filename == "" or not allowed_file(file.filename):
            return "Invalid file", 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
        except Exception as e:
            return f"Error processing CSV: {e}", 500
        try:
            df = preprocess_support_data(df)
        except Exception as e:
            return f"Preprocessing error: {e}", 500
        try:
            threshold = float(request.form.get("threshold", "0.85"))
        except ValueError:
            threshold = 0.85
        recommendations = get_support_recommendations(df, threshold)
        recommendations = sorted(recommendations, key=lambda x: x['Similarity Score'], reverse=True)
    return render_template("production_support_triage.html", recommendations=recommendations)

@app.route("/save_creds/<provider>", methods=["POST"])
def save_creds(provider):
    data = request.get_json()
    if provider.lower() == "jira":
        filename = "JiraCreds.json"
    elif provider.lower() == "aha":
        filename = "AhaCreds.json"
    else:
        return jsonify({"error": "Invalid provider"}), 400
    try:
        with open(filename, "w") as f:
            json.dump(data, f)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "Credentials saved"}), 200

@app.route("/load_creds/<provider>", methods=["GET"])
def load_creds(provider):
    if provider.lower() == "jira":
        filename = "JiraCreds.json"
    elif provider.lower() == "aha":
        filename = "AhaCreds.json"
    else:
        return jsonify({"error": "Invalid provider"}), 400
    if not os.path.exists(filename):
        return jsonify({"error": "No credentials file found"}), 404
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(data), 200

@app.route("/merge", methods=["POST"])
def merge_ideas():
    data = request.get_json()
    needs_idea = data.get("needs_idea")
    target_idea = data.get("target_idea")
    response = {"status": "success", "message": f"Idea {needs_idea} merged into {target_idea}."}
    return jsonify(response), 200

# New endpoint for support tickets merge simulation
@app.route("/merge_support", methods=["POST"])
def merge_support():
    data = request.get_json()
    ticket_id = data.get("ticket_id")
    candidate_id = data.get("candidate_id")
    response = {"status": "success", "message": f"Ticket {ticket_id} merged into {candidate_id}."}
    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)
