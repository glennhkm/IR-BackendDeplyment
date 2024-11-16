import re
import numpy as np
from pymongo import MongoClient
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

try:
    client = MongoClient("mongodb://localhost:27017",)
    db = client["local"]
    tf_idf_collection = db['tf_idf']
    feature_collection = db['feature']
    news_data_collection = db['news_data']
    
    feature_documents = list(feature_collection.find())
    
    if not feature_documents:
        raise Exception("No feature documents found in collections")
    
    print(f"Loaded {len(feature_documents)} feature documents")
    
    feature_names = feature_documents[0]['feature_names']
    vectorizer = TfidfVectorizer(vocabulary=feature_names)
    dummy_doc = ' '.join(feature_names)    
    vectorizer.fit([dummy_doc])
    
    stemmer_factory = StemmerFactory()
    stopword_factory = StopWordRemoverFactory()
    stemmer = stemmer_factory.create_stemmer()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    print("Application initialized successfully")
    print(f"Vocabulary size: {len(feature_names)}")
    
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

def extract_slug(path):
    parts = path.strip('/').split('/')
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return None, parts[-1]
    
def preprocess_text(text):
    text = re.sub(r'[^\w\s]|[\d]', ' ', text)
    text = text.lower()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    tokens = text.split()     
    return tokens
    
@app.route("/search", methods=["POST"])    
def search():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"]
        category = data.get("category", "").title()
        
        query_tokens = preprocess_text(query)
        query_indices = [feature_names.index(token) for token in query_tokens if token in feature_names]
        
        if not query_indices:
            return jsonify({"error": "No valid tokens found in query"}), 400
        
        pipeline = [
            {    
            "$match": {
                "$expr": {
                "$gt": [
                    {
                    "$size": {
                        "$filter": {
                        "input": query_indices,
                        "as": "index",
                        "cond": { "$ne": [{ "$arrayElemAt": ["$tfidf_vector", "$$index"] }, 0] }
                        }
                    }
                    },
                    0
                ]
                }
            }
            },
            {
            "$lookup": {
                "from": "news_data",
                "localField": "Document_id",
                "foreignField": "_id",
                "as": "news_data"
            }
            },
            { "$unwind": "$news_data" },
            { "$match": {"news_data.Kategori": category} } if category != "All" else None,
            { "$project": { "news_data._id": 0, "news_data.Isi Berita": 0 } }
        ]
        
        # pipeline = [
        #     {    
        #         "$match": {
        #             "$expr": {
        #                 "$eq": [
        #                     {
        #                         "$size": {
        #                             "$filter": {
        #                                 "input": query_indices,
        #                                 "as": "index",
        #                                 "cond": { "$ne": [{ "$arrayElemAt": ["$tfidf_vector", "$$index"] }, 0] }
        #                             }
        #                         }
        #                     },
        #                     len(query_indices)
        #                 ]
        #             }
        #         }
        #     },
        #     {
        #         "$lookup": {
        #             "from": "news_data",
        #             "localField": "Document_id",
        #             "foreignField": "_id",
        #             "as": "news_data"
        #         }
        #     },
        #     { "$unwind": "$news_data" },
        #     { "$match": {"news_data.Kategori": category} } if category != "All" else None,
        #     { "$project": { "news_data._id": 0, "news_data.Isi Berita": 0 } }
        # ]

        pipeline = [stage for stage in pipeline if stage is not None]

        tf_idf_documents = list(tf_idf_collection.aggregate(pipeline))
        
        if not tf_idf_documents:
            return jsonify([]), 200
        
        print(f"Processing search query: {query} for category: {category}")
                            
        tfidf_vectors = [doc['tfidf_vector'] for doc in tf_idf_documents]
        tfidf_matrix_filtered = np.array(tfidf_vectors)
        
        query_str = ' '.join(query_tokens)
        query_vector = vectorizer.transform([query_str])        
        query_vector_dense = query_vector.toarray().T
        
        scores = np.dot(tfidf_matrix_filtered, query_vector_dense).flatten()        
        
        top_k = min(100, len(scores))
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:
                news_data = tf_idf_documents[idx].get("news_data", {})
                results.append({
                    "title": news_data.get("Judul", "No title"),
                    "author": news_data.get("Pengarang", "No author"),
                    "summary": news_data.get("Ringkasan", "No summary"),
                    "url": news_data.get("Url", "No Url"),
                    "slug": news_data.get("Slug", "No slug"),
                    "date": news_data.get("Tanggal", "No date"),
                    "category": news_data.get("Kategori", "No category"),
                    "score": float(scores[idx])
                })
        
        print(f"Found {len(results)} results for category: {category}")
        return jsonify(results)
    
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/news/<path:slug>", methods=["GET"])
def get(slug):  
    try:
        category, article_slug = extract_slug(slug)
        news_data = news_data_collection.find_one({
            "Slug": article_slug,
            "Kategori": category.title()
        })
        if news_data:
            news_data['_id'] = str(news_data['_id'])
            return jsonify(news_data)
        else:
            return jsonify({"error": "No news data found for the given slug"}), 404
    except Exception as e:
        print(f"Error during search: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "API is running",
        "vocabulary_size": len(feature_names),
    })

if __name__ == "__main__":
    try:
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")