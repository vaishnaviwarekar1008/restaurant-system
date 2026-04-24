from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load Model Data
print("Loading model data...")
model_path = os.path.join(os.path.dirname(__file__), 'restaurant.pkl')
try:
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    cosine_sim = model_data['similarity_matrix']
    indices = model_data['indices']
    df_restaurants = model_data['restaurant_data']
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}. Please ensure build_model.py has completed successfully.")
    cosine_sim, indices, df_restaurants = None, None, pd.DataFrame()

def get_recommendations(name, cosine_sim=cosine_sim):
    if cosine_sim is None or df_restaurants.empty:
        return []
        
    # Ensure name exists
    if name not in indices:
        return []
    
    idx = indices[name]
    # In case there are multiple matching indices, take the first one
    if isinstance(idx, pd.Series):
        idx = idx.iloc[0]
        
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] # Top 10 recommendations
    
    restaurant_indices = [i[0] for i in sim_scores]
    recommendations = df_restaurants.iloc[restaurant_indices].copy()
    
    # Store similarity scores for display
    # The sim_scores is a list of tuples: (index, score)
    scores = {i[0]: i[1] for i in sim_scores}
    
    # Format data for UI
    results = []
    for idx_row, row in recommendations.iterrows():
        # Clean restaurant name (handle any remaining encoding issues)
        clean_name = str(row['name']).strip()
        score_val = round(scores[idx_row] * 100) # Convert 0.0-1.0 to 0-100%
        
        results.append({
            'name': clean_name,
            'cuisines': str(row['cuisines']),
            'location': str(row['location']),
            'rate': round(float(row['rate']), 1) if pd.notnull(row['rate']) else 'N/A',
            'cost': str(row['approx_cost(for two people)']),
            'score': score_val
        })
    return results

@app.route('/')
def home():
    if df_restaurants.empty:
        restaurant_names = []
    else:
        restaurant_names = sorted(df_restaurants['name'].unique().tolist())
    return render_template('index.html', restaurants=restaurant_names)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        restaurant_name = request.form['restaurant_name']
        recommendations = get_recommendations(restaurant_name)
        
        restaurant_names = [] if df_restaurants.empty else sorted(df_restaurants['name'].unique().tolist())
        
        return render_template('index.html', 
                               restaurants=restaurant_names, 
                               selected_restaurant=restaurant_name,
                               recommendations=recommendations,
                               show_results=True)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
