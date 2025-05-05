This Flask application uses LightGBM machine learning models to predict which customers are most likely to purchase airline tickets, both in general and for specific destinations.


- Predicts general purchase likelihood for all customers
- Predicts destination-specific purchase likelihood
- Filters customers by minimum score threshold
- Special handling for new routes (Hawaii) based on purchase patterns for similar destinations

How We Predict Who Will Buy

- Purchase Recency: Customers who bought tickets more recently receive higher scores, as they're more likely to book again soon
- Purchase Frequency: Frequent flyers get higher scores - someone who flies 4 times a year is more likely to book again than someone who flies once a year
- Travel Spending: Higher-spending customers (those buying business or first-class tickets) receive higher scores as they demonstrate stronger buying power
- Destination Loyalty: Customers who repeatedly visit the same destination get higher scores when predicting likelihood to return to that destination
- Travel Patterns: We analyze how regularly customers travel (e.g., seasonal patterns, average days between trips) to identify those due for another booking
- Destination Similarity: For new routes like Hawaii, we identify customers who travel to similar destinations (tropical locations, beach destinations, long-haul flights)
- Ticket Type Preference: Customers' history of economy, business, or first-class purchases helps determine their price sensitivity and travel preferences

This system helps target your marketing efforts on customers most likely to respond positively, improving conversion rates and maximizing marketing ROI.

