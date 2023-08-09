import praw
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import BertTokenizer, BertModel
import torch

# Step 1: Data Collection
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

def get_subreddit_posts(subreddit_name, num_posts=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for post in subreddit.hot(limit=num_posts):
        posts.append(post.title + " " + post.selftext)
    return posts

# Step 2: Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

# Step 3: Feature Extraction
def extract_features(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data)
    return tfidf_matrix, tfidf_vectorizer

# Step 4: Topic Modeling
def perform_topic_modeling(tfidf_matrix, num_topics):
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(tfidf_matrix)
    return lda_model

# Step 5: User Profile Generation
def get_user_profile(user_interactions):
    user_profile = " ".join([post for subreddit in user_interactions.values() for post in subreddit])
    return user_profile

# Step 6: Cosine Similarity Calculation
def get_recommendations(user_profile, lda_model, tfidf_vectorizer):
    user_profile_tfidf = tfidf_vectorizer.transform([user_profile])
    user_topic_distribution = lda_model.transform(user_profile_tfidf)
    similarity_scores = cosine_similarity(user_topic_distribution, lda_model.components_)
    return similarity_scores

# Step 8: Implementing Transformer for Encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def encode_subreddit_posts(posts):
    encoded_posts = tokenizer.batch_encode_plus(
        posts,
        add_special_tokens=True,
        pad_to_max_length=True,
        return_tensors='pt',
        max_length=128
    )
    with torch.no_grad():
        model_outputs = model(**encoded_posts)
    return model_outputs.pooler_output

# Step 9: User Profile Generation with Transformer
def get_user_profile_embedding(user_interactions, encoded_data):
    user_posts = [post for subreddit in user_interactions.values() for post in subreddit]
    user_encoded_data = encode_subreddit_posts(user_posts)
    user_profile_embedding = torch.mean(user_encoded_data, dim=0, keepdim=True)
    return user_profile_embedding

def get_recommendations_with_transformer(user_profile_embedding, encoded_data):
    similarity_scores = torch.cosine_similarity(user_profile_embedding, encoded_data)
    return similarity_scores

# Example usage
subreddit_name = 'AskReddit'
num_posts = 1000
data = get_subreddit_posts(subreddit_name, num_posts)
cleaned_data = [preprocess_text(text) for text in data]

# Steps 3 and 4
tfidf_matrix, tfidf_vectorizer = extract_features(cleaned_data)
num_topics = 10
lda_model = perform_topic_modeling(tfidf_matrix, num_topics)

# Step 5 (User Profile Generation)
user_interactions = {
    'user1': ['AskReddit', 'funny', 'pics'],
    'user2': ['AskReddit', 'worldnews', 'gaming'],
    # Add more users and their subreddit interactions
}

user_profile = get_user_profile(user_interactions)

# Steps 6 and 7 (Cosine Similarity Calculation and Recommendation Generation)
recommendations = get_recommendations(user_profile, lda_model, tfidf_vectorizer)

# Step 8 (Encoding subreddit posts with Transformer)
encoded_data = encode_subreddit_posts(cleaned_data)

# Step 9 (User Profile Generation with Transformer and Recommendation Generation)
user_profile_embedding = get_user_profile_embedding(user_interactions, encoded_data)
recommendations_transformer = get_recommendations_with_transformer(user_profile_embedding, encoded_data)

print("LDA-based Recommendations:")
print(recommendations)

print("Transformer-based Recommendations:")
print(recommendations_transformer)
