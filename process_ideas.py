#!/usr/bin/env python3
"""
This script reads the CSV file containing idea records, filters out records with the status “Needs Review”
and for each such idea recommends a similar idea (from those not in "Needs Review" status) based on text similarity.

Approach:
1. Read the CSV file using pandas.
2. Create a combined text field from “Idea name” and “Idea description”.
3. Split records into two groups:
   • ideas_needs_review: ideas with status exactly “Needs Review” (case-insensitive).
   • ideas_others: ideas with any other status.
4. Use TfidfVectorizer to vectorize the combined text.
5. For each idea in the “Needs Review” group, compute cosine similarities with all ideas in the other group, 
   and pick the most similar candidate.
6. Output the recommendations to the console.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    return df

def preprocess_data(df):
    # Ensure the required columns exist
    required_columns = ['Idea reference', 'Idea name', 'Idea status', 'Idea description']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            sys.exit(1)
    # Create a new column 'combined_text'
    # Replace NaN values with an empty string
    df['Idea name'] = df['Idea name'].fillna('')
    df['Idea description'] = df['Idea description'].fillna('')
    df['combined_text'] = df['Idea name'] + " " + df['Idea description']
    # Normalize status strings to lower-case for comparison
    df['Idea status_lower'] = df['Idea status'].str.lower().str.strip()
    return df

def get_recommendations(df):
    # Filter ideas with "needs review" and others
    # We are checking primarily for exact match "needs review" in lower case.
    needs_review_df = df[df['Idea status_lower'] == 'needs review'].copy()
    others_df = df[df['Idea status_lower'] != 'needs review'].copy()

    if needs_review_df.empty:
        print("No ideas with status 'Needs Review' found.")
        return

    if others_df.empty:
        print("No other ideas available for recommendation (all are 'Needs Review').")
        return

    # Vectorize the combined text for both groups
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer on the union of texts from both groups
    corpus = pd.concat([needs_review_df['combined_text'], others_df['combined_text']])
    vectorizer.fit(corpus)

    # Transform texts
    needs_vectors = vectorizer.transform(needs_review_df['combined_text'])
    others_vectors = vectorizer.transform(others_df['combined_text'])
    
    # Compute cosine similarity between each needs_review idea and all others
    similarity_matrix = cosine_similarity(needs_vectors, others_vectors)

    recommendations = []
    threshold = 0.85
    for idx, similarities in enumerate(similarity_matrix):
        # Get the index of the most similar idea in others_df
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

def main():
    csv_path = "aha_list_ideas_250409131454.csv"
    print("Loading data from CSV...")
    df = load_data(csv_path)
    df = preprocess_data(df)
    print("Computing recommendations...")
    recs = get_recommendations(df)
    if recs:
        print("\nRecommended Merges (for ideas in 'Needs Review'):")
        for rec in recs:
            print(f"  - Idea {rec['Needs Review Idea']} -> Merge with Idea {rec['Recommended Merge Candidate']} (Score: {rec['Similarity Score']:.4f})")
        # Write recommendations to CSV
        rec_df = pd.DataFrame(recs)
        rec_df.to_csv("recommended_Merges.csv", index=False)
        print("\nRecommendations saved to recommended_Merges.csv")

if __name__ == "__main__":
    main()
