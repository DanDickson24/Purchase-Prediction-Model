from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def load_customer_data():
    with open('customers.json', 'r') as file:
        data = json.load(file)
    return data

def create_features_df(customer_data):
    current_date = datetime.now()
    
    customers = []
    purchases = []

    for customer in customer_data['customers']:
        customer_id = customer['id']
        name = customer['name']
        address = customer['address']
        customers.append({
            'customer_id': customer_id,
            'name': name,
            'address': address
        })
        
        for purchase in customer['purchases']:
            purchase_date = datetime.strptime(purchase['date'], '%Y-%m-%d')
            destination = purchase['destination']
            ticket_type = purchase['ticket_type']
            price = purchase['price']
            
            purchases.append({
                'customer_id': customer_id,
                'purchase_date': purchase_date,
                'destination': destination,
                'ticket_type': ticket_type,
                'price': price
            })
    
    customers_df = pd.DataFrame(customers)
    purchases_df = pd.DataFrame(purchases)

    customer_features = []
    for customer_id in customers_df['customer_id'].unique():
        customer_purchases = purchases_df[purchases_df['customer_id'] == customer_id]
        first_purchase = customer_purchases['purchase_date'].min()
        last_purchase = customer_purchases['purchase_date'].max()
        total_purchases = len(customer_purchases)
        days_since_first = (current_date - first_purchase).days
        days_since_last = (current_date - last_purchase).days

        if total_purchases > 1:
            purchase_span = (last_purchase - first_purchase).days
            avg_days_between = purchase_span / (total_purchases - 1)
        else:
            purchase_span = 0
            avg_days_between = 365  
        
        destinations = customer_purchases['destination'].value_counts()
        most_frequent_destination = destinations.index[0] if not destinations.empty else None
        num_unique_destinations = len(destinations)
        
        ticket_types = customer_purchases['ticket_type'].value_counts()
        most_frequent_ticket = ticket_types.index[0] if not ticket_types.empty else None
        
        avg_price = customer_purchases['price'].mean()
        max_price = customer_purchases['price'].max()
        
        dest_metrics = {}
        for dest in ['New York', 'Bangkok', 'Orlando', 'Los Angeles', 'Miami']:
            dest_purchases = customer_purchases[customer_purchases['destination'] == dest]
            dest_count = len(dest_purchases)
            if dest_count > 0:
                last_dest_purchase = dest_purchases['purchase_date'].max()
                days_since_last_dest = (current_date - last_dest_purchase).days
                avg_dest_price = dest_purchases['price'].mean()
            else:
                days_since_last_dest = 9999 
                avg_dest_price = 0
                
            dest_metrics[f'trips_to_{dest.replace(" ", "_")}'] = dest_count
            dest_metrics[f'days_since_{dest.replace(" ", "_")}'] = days_since_last_dest
            dest_metrics[f'avg_price_{dest.replace(" ", "_")}'] = avg_dest_price
        
        feature_dict = {
            'customer_id': customer_id,
            'days_since_first': days_since_first,
            'days_since_last': days_since_last,
            'total_purchases': total_purchases,
            'purchase_span': purchase_span,
            'avg_days_between': avg_days_between,
            'num_unique_destinations': num_unique_destinations,
            'most_frequent_destination': most_frequent_destination,
            'most_frequent_ticket': most_frequent_ticket,
            'avg_price': avg_price,
            'max_price': max_price,
            **dest_metrics
        }
        
        customer_features.append(feature_dict)
    
    features_df = pd.DataFrame(customer_features)
    features_df = features_df.merge(customers_df, on='customer_id')
    
    return features_df

def calculate_general_scores(features_df):
    df = features_df.copy()
    recency_weight = -0.05  
    frequency_weight = 10  
    avg_days_weight = -0.1  
    price_weight = 0.01    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[['days_since_last', 'total_purchases', 'avg_days_between', 'avg_price']])
    scaled_df = pd.DataFrame(scaled_features, columns=['days_since_last_scaled', 'total_purchases_scaled', 
                                                       'avg_days_between_scaled', 'avg_price_scaled'])
    
    df['purchase_score'] = (
        scaled_df['days_since_last_scaled'] * recency_weight * -1 +  
        scaled_df['total_purchases_scaled'] * frequency_weight +
        scaled_df['avg_days_between_scaled'] * avg_days_weight * -1 +  
        scaled_df['avg_price_scaled'] * price_weight
    )
    
    df['purchase_score'] += df['num_unique_destinations'] * 0.5
    min_score = df['purchase_score'].min()
    max_score = df['purchase_score'].max()
    df['purchase_score'] = ((df['purchase_score'] - min_score) / 
                           (max_score - min_score) * 100).clip(0, 100)
    df['purchase_score'] = df['purchase_score'].round(1)
    
    return df[['customer_id', 'name', 'purchase_score']]

def calculate_destination_scores(features_df, destination):
    dest_col = destination.replace(" ", "_")
    
    df = features_df.copy()
    recency_weight = 25      
    frequency_weight = 15   
    price_weight = 10       
    max_days = df['days_since_last'].max()
    if max_days > 0:
        df['recency_score'] = (max_days - df['days_since_last']) / max_days
    else:
        df['recency_score'] = 0
        
    max_purchases = df['total_purchases'].max()
    if max_purchases > 0:
        df['frequency_score'] = df['total_purchases'] / max_purchases
    else:
        df['frequency_score'] = 0
        
    max_price = df['avg_price'].max()
    if max_price > 0:
        df['price_score'] = df['avg_price'] / max_price
    else:
        df['price_score'] = 0
    
    df['base_score'] = (
        df['recency_score'] * recency_weight +
        df['frequency_score'] * frequency_weight +
        df['price_score'] * price_weight
    )
    

    if destination in ['New York', 'Bangkok', 'Orlando', 'Los Angeles', 'Miami']:
        dest_trips_col = f'trips_to_{dest_col}'
        dest_days_col = f'days_since_{dest_col}'
        max_dest_trips = df[dest_trips_col].max()
        if max_dest_trips > 0:
            df['dest_freq_score'] = df[dest_trips_col] / max_dest_trips
        else:
            df['dest_freq_score'] = 0
            
        df['dest_recency_score'] = 0.1  
        visited_mask = df[dest_days_col] < 9999
        if visited_mask.any():
            max_dest_days = df.loc[visited_mask, dest_days_col].max()
            if max_dest_days > 0:
                df.loc[visited_mask, 'dest_recency_score'] = (max_dest_days - df.loc[visited_mask, dest_days_col]) / max_dest_days
        

        df[f'{dest_col}_score'] = df['base_score'] * 0.6 + (
            df['dest_freq_score'] * 25 +
            df['dest_recency_score'] * 15
        )
    
    elif destination == 'Hawaii':
        similar_dests = {
            'Miami': 30,       
            'Los Angeles': 25,   
            'Bangkok': 25,      
            'Orlando': 20     
        }

        df['similarity_score'] = 0
        for similar_dest, weight in similar_dests.items():
            similar_col = similar_dest.replace(" ", "_")
            trips_col = f'trips_to_{similar_col}'
            max_similar_trips = df[trips_col].max()
            if max_similar_trips > 0:
                df[f'{similar_col}_norm'] = df[trips_col] / max_similar_trips
            else:
                df[f'{similar_col}_norm'] = 0
                
            df['similarity_score'] += df[f'{similar_col}_norm'] * weight
            
        max_similarity = df['similarity_score'].max()
        if max_similarity > 0:
            df['similarity_score'] = df['similarity_score'] / max_similarity * 100
            
        df[f'{dest_col}_score'] = df['base_score'] * 0.6 + df['similarity_score'] * 0.4
    
    else:
        df[f'{dest_col}_score'] = df['base_score']
    
    min_score = df[f'{dest_col}_score'].min()
    max_score = df[f'{dest_col}_score'].max()
    df[f'{dest_col}_score'] = ((df[f'{dest_col}_score'] - min_score) / 
                              (max_score - min_score) * 100).clip(0, 100)
    
    df[f'{dest_col}_score'] = df[f'{dest_col}_score'].round(1)

    return df[['customer_id', 'name', f'{dest_col}_score']]

def get_customers_above_threshold(predictions_df, threshold=70, score_col='purchase_score'):

    high_score_customers = predictions_df[predictions_df[score_col] >= threshold]
    
    high_score_customers = high_score_customers.sort_values(by=score_col, ascending=False)
    
    return high_score_customers

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction_type = data.get('prediction_type', 'general')
    threshold = float(data.get('threshold', 70))
    destination = data.get('destination', None)
    
    customer_data = load_customer_data()
    
    features_df = create_features_df(customer_data)
    
    if prediction_type == 'general':
        predictions = calculate_general_scores(features_df)
        
        high_score_customers = get_customers_above_threshold(predictions, threshold, 'purchase_score')

        result = [{
            'id': int(row['customer_id']),
            'name': row['name'],
            'score': float(row['purchase_score'])
        } for _, row in high_score_customers.iterrows()]
        
        return jsonify({
            'prediction_type': 'General Purchase Likelihood',
            'threshold': threshold,
            'customers': result
        })
    
    elif prediction_type == 'destination':
        if not destination:
            return jsonify({'error': 'Destination is required for destination-specific prediction'})
        
        dest_col = destination.replace(" ", "_")
        predictions = calculate_destination_scores(features_df, destination)
        
        high_score_customers = get_customers_above_threshold(predictions, threshold, f'{dest_col}_score')
        
        result = [{
            'id': int(row['customer_id']),
            'name': row['name'],
            'score': float(row[f'{dest_col}_score'])
        } for _, row in high_score_customers.iterrows()]
        
        return jsonify({
            'prediction_type': f'{destination} Purchase Likelihood',
            'threshold': threshold,
            'customers': result
        })
    
    else:
        return jsonify({'error': 'Invalid prediction type'})

if __name__ == '__main__':
    app.run(debug=True)