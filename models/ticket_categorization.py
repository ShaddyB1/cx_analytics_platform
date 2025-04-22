#!/usr/bin/env python3
"""
Ticket Categorization Model

This module implements an NLP-based model for categorizing and prioritizing
customer service tickets based on content analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
import os
import pickle
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources (uncomment first time)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class TicketCategorizer:
    """
    NLP-based model for categorizing customer service tickets
    and determining their priority.
    """
    
    def __init__(
        self,
        use_bert=False,  # If True, use BERT embeddings, else use TF-IDF
        model_path=None
    ):
        """
        Initialize the ticket categorizer.
        
        Parameters:
        -----------
        use_bert : bool
            Whether to use BERT embeddings (requires transformers package)
        model_path : str
            Path to load a saved model from (if None, creates a new model)
        """
        self.use_bert = use_bert
        self.type_model = None
        self.priority_model = None
        self.type_encoder = None  # For encoding ticket types
        self.priority_encoder = None  # For encoding priorities
        self.is_fitted = False
        self.metrics = {}
        
        # Text preprocessing tools
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load a saved model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
    def _preprocess_text(self, text):
        """
        Preprocess text for NLP model.
        
        Parameters:
        -----------
        text : str
            Text content to preprocess
            
        Returns:
        --------
        str
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Tokenize
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        
        # Rejoin
        return ' '.join(tokens)
    
    def _prepare_features(self, data, text_column='content', customer_columns=None):
        """
        Prepare features for the model from raw data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw data containing ticket information
        text_column : str
            Name of the column containing ticket text
        customer_columns : list
            List of customer attribute columns to include
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with prepared features
        """
        # Ensure text column exists
        if text_column not in data.columns:
            raise ValueError(f"Text column '{text_column}' not found in data")
        
        # Preprocess text
        logger.info(f"Preprocessing ticket text from column '{text_column}'")
        data['processed_text'] = data[text_column].apply(self._preprocess_text)
        
        # Include customer features if provided
        if customer_columns:
            valid_columns = [col for col in customer_columns if col in data.columns]
            if valid_columns:
                logger.info(f"Including customer attributes: {valid_columns}")
                return data[['processed_text'] + valid_columns]
            
        return data[['processed_text']]
    
    def _extract_emergency_keywords(self, text):
        """
        Extract keywords that might indicate urgency.
        
        Parameters:
        -----------
        text : str
            Text to analyze
            
        Returns:
        --------
        float
            Urgency score based on keyword presence
        """
        if not isinstance(text, str):
            return 0.0
        
        # Convert to lowercase for case-insensitive matching
        text = text.lower()
        
        # Define urgency indicators with weights
        urgency_indicators = {
            'urgent': 1.0,
            'emergency': 1.0,
            'immediately': 0.8,
            'critical': 0.9,
            'asap': 0.7,
            'right now': 0.7,
            'deadline': 0.6,
            'important': 0.5,
            'stuck': 0.4,
            'broken': 0.4,
            'error': 0.3,
            'issue': 0.2,
            'problem': 0.2,
            'help': 0.1
        }
        
        # Calculate weighted score based on occurrences
        score = 0.0
        for word, weight in urgency_indicators.items():
            if word in text:
                score += weight
        
        # Cap at 1.0
        return min(1.0, score)
    
    def _build_tfidf_pipeline(self):
        """
        Build a TF-IDF based classification pipeline.
        
        Returns:
        --------
        sklearn.pipeline.Pipeline
            Classification pipeline
        """
        return Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                min_df=2,
                max_df=0.8,
                ngram_range=(1, 2)
            )),
            ('classifier', OneVsRestClassifier(
                LogisticRegression(
                    solver='liblinear',
                    C=1.0,
                    max_iter=1000
                )
            ))
        ])
    
    def _build_bert_pipeline(self):
        """
        Build a BERT embedding based classification pipeline.
        
        Returns:
        --------
        sklearn.pipeline.Pipeline
            Classification pipeline
        """
        try:
            from transformers import BertTokenizer, BertModel
            import torch
            from sklearn.base import BaseEstimator, TransformerMixin
            
            class BertEmbedder(BaseEstimator, TransformerMixin):
                def __init__(self, model_name='bert-base-uncased'):
                    self.model_name = model_name
                    self.tokenizer = BertTokenizer.from_pretrained(model_name)
                    self.model = BertModel.from_pretrained(model_name)
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.model.to(self.device)
                    self.model.eval()
                
                def transform(self, texts):
                    embeddings = []
                    for text in texts:
                        # Truncate long texts
                        if len(text.split()) > 100:
                            text = ' '.join(text.split()[:100])
                            
                        inputs = self.tokenizer(
                            text,
                            return_tensors='pt',
                            padding=True,
                            truncation=True,
                            max_length=128
                        ).to(self.device)
                        
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                            # Use the CLS token embedding as the text representation
                            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                            embeddings.append(embedding)
                    
                    return np.array(embeddings)
                
                def fit(self, X, y=None):
                    return self
            
            return Pipeline([
                ('bert_embedder', BertEmbedder()),
                ('classifier', OneVsRestClassifier(
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000
                    )
                ))
            ])
            
        except ImportError:
            logger.warning("Transformers package not found. Falling back to TF-IDF.")
            self.use_bert = False
            return self._build_tfidf_pipeline()
    
    def fit(self, ticket_data, text_column='content', type_column='type', 
            priority_column='priority', customer_data=None, customer_id_column='customer_id',
            train_size=0.8):
        """
        Fit the ticket categorization models.
        
        Parameters:
        -----------
        ticket_data : pandas.DataFrame
            DataFrame containing ticket information
        text_column : str
            Name of the column containing ticket text
        type_column : str
            Name of the column containing ticket type
        priority_column : str
            Name of the column containing priority
        customer_data : pandas.DataFrame
            DataFrame containing customer information (optional)
        customer_id_column : str
            Name of the column containing customer ID in both dataframes
        train_size : float
            Fraction of data to use for training (0 to 1)
            
        Returns:
        --------
        self
        """
        # Ensure required columns exist
        required_columns = [text_column, type_column, priority_column]
        missing_columns = [col for col in required_columns if col not in ticket_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Fitting ticket categorization models on {len(ticket_data)} tickets")
        
        # Merge customer data if provided
        if customer_data is not None:
            if customer_id_column not in ticket_data.columns:
                raise ValueError(f"Customer ID column '{customer_id_column}' not found in ticket data")
            if customer_id_column not in customer_data.columns:
                raise ValueError(f"Customer ID column '{customer_id_column}' not found in customer data")
                
            logger.info(f"Merging customer data with ticket data using '{customer_id_column}'")
            data = pd.merge(
                ticket_data,
                customer_data,
                on=customer_id_column,
                how='left'
            )
            
            # Customer attributes to include as features
            customer_columns = ['account_type', 'ticket_frequency']
        else:
            data = ticket_data.copy()
            customer_columns = None
        
        # Prepare features
        X = self._prepare_features(data, text_column, customer_columns)
        
        # Extract urgency keywords
        X['urgency_score'] = data[text_column].apply(self._extract_emergency_keywords)
        
        # Prepare target variables
        y_type = data[type_column]
        y_priority = data[priority_column]
        
        # Encode target variables
        self.type_encoder = LabelEncoder()
        self.type_encoder.fit(y_type)
        y_type_encoded = self.type_encoder.transform(y_type)
        
        self.priority_encoder = LabelEncoder()
        self.priority_encoder.fit(y_priority)
        y_priority_encoded = self.priority_encoder.transform(y_priority)
        
        # Train-test split
        X_train, X_test, y_type_train, y_type_test, y_priority_train, y_priority_test = train_test_split(
            X, y_type_encoded, y_priority_encoded, train_size=train_size, random_state=42
        )
        
        # Build models
        logger.info(f"Building {'BERT' if self.use_bert else 'TF-IDF'} based models")
        
        if self.use_bert:
            self.type_model = self._build_bert_pipeline()
            self.priority_model = self._build_bert_pipeline()
        else:
            self.type_model = self._build_tfidf_pipeline()
            self.priority_model = self._build_tfidf_pipeline()
        
        # Train type model
        logger.info("Training ticket type classification model")
        self.type_model.fit(X_train['processed_text'], y_type_train)
        
        # Train priority model with urgency score
        logger.info("Training ticket priority classification model")
        # Combine text features with urgency score for priority prediction
        X_train_priority = pd.DataFrame({
            'processed_text': X_train['processed_text'],
            'urgency_score': X_train['urgency_score']
        })
        X_test_priority = pd.DataFrame({
            'processed_text': X_test['processed_text'],
            'urgency_score': X_test['urgency_score']
        })
        
        self.priority_model.fit(X_train_priority, y_priority_train)
        
        # Evaluate models
        logger.info("Evaluating models")
        
        # Ticket type evaluation
        y_type_pred = self.type_model.predict(X_test['processed_text'])
        type_accuracy = accuracy_score(y_type_test, y_type_pred)
        type_f1 = f1_score(y_type_test, y_type_pred, average='weighted')
        
        # Priority evaluation
        y_priority_pred = self.priority_model.predict(X_test_priority)
        priority_accuracy = accuracy_score(y_priority_test, y_priority_pred)
        priority_f1 = f1_score(y_priority_test, y_priority_pred, average='weighted')
        
        # Store metrics
        self.metrics = {
            'type_accuracy': type_accuracy,
            'type_f1': type_f1,
            'priority_accuracy': priority_accuracy,
            'priority_f1': priority_f1
        }
        
        logger.info(f"Type model accuracy: {type_accuracy:.4f}, F1: {type_f1:.4f}")
        logger.info(f"Priority model accuracy: {priority_accuracy:.4f}, F1: {priority_f1:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, tickets, text_column='content', customer_data=None, 
                customer_id_column='customer_id'):
        """
        Predict ticket types and priorities.
        
        Parameters:
        -----------
        tickets : pandas.DataFrame or str
            DataFrame containing tickets or a single ticket text
        text_column : str
            Name of the column containing ticket text
        customer_data : pandas.DataFrame
            DataFrame containing customer information (optional)
        customer_id_column : str
            Name of the column containing customer ID
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with predicted types and priorities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle single string input
        if isinstance(tickets, str):
            tickets = pd.DataFrame({text_column: [tickets]})
        
        # Merge customer data if provided
        if customer_data is not None:
            if customer_id_column in tickets.columns and customer_id_column in customer_data.columns:
                data = pd.merge(
                    tickets,
                    customer_data,
                    on=customer_id_column,
                    how='left'
                )
                customer_columns = ['account_type', 'ticket_frequency']
            else:
                data = tickets.copy()
                customer_columns = None
        else:
            data = tickets.copy()
            customer_columns = None
        
        # Prepare features
        X = self._prepare_features(data, text_column, customer_columns)
        
        # Extract urgency keywords
        X['urgency_score'] = data[text_column].apply(self._extract_emergency_keywords)
        
        # Make predictions
        logger.info(f"Making predictions for {len(tickets)} tickets")
        
        # Ticket type predictions
        type_probs = self.type_model.predict_proba(X['processed_text'])
        type_preds = np.argmax(type_probs, axis=1)
        type_preds = self.type_encoder.inverse_transform(type_preds)
        
        # Priority predictions
        X_priority = pd.DataFrame({
            'processed_text': X['processed_text'],
            'urgency_score': X['urgency_score']
        })
        priority_probs = self.priority_model.predict_proba(X_priority)
        priority_preds = np.argmax(priority_probs, axis=1)
        priority_preds = self.priority_encoder.inverse_transform(priority_preds)
        
        # Create result DataFrame
        results = pd.DataFrame({
            'predicted_type': type_preds,
            'predicted_priority': priority_preds,
            'type_confidence': np.max(type_probs, axis=1),
            'priority_confidence': np.max(priority_probs, axis=1),
            'urgency_score': X['urgency_score']
        })
        
        # Add ticket ID if available
        if 'ticket_id' in tickets.columns:
            results['ticket_id'] = tickets['ticket_id']
        
        return results
    
    def save_model(self, path):
        """
        Save the trained model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        model_data = {
            'type_model': self.type_model,
            'priority_model': self.priority_model,
            'type_encoder': self.type_encoder,
            'priority_encoder': self.priority_encoder,
            'use_bert': self.use_bert,
            'metrics': self.metrics,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load a trained model from disk.
        
        Parameters:
        -----------
        path : str
            Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        model_data = joblib.load(path)
        
        self.type_model = model_data['type_model']
        self.priority_model = model_data['priority_model']
        self.type_encoder = model_data['type_encoder']
        self.priority_encoder = model_data['priority_encoder']
        self.use_bert = model_data['use_bert']
        self.metrics = model_data['metrics']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"Model was saved at: {model_data.get('saved_at', 'unknown')}")
        
    def get_model_info(self):
        """
        Get information about the fitted model.
        
        Returns:
        --------
        dict
            Dictionary containing model information
        """
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        return {
            "status": "Fitted",
            "metrics": self.metrics,
            "ticket_types": self.type_encoder.classes_.tolist(),
            "priority_levels": self.priority_encoder.classes_.tolist(),
            "model_type": "BERT" if self.use_bert else "TF-IDF"
        }


# Example usage
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.generate_sample_data import generate_tickets, generate_agents, generate_customers
    
    # Generate sample data
    print("Generating sample data...")
    agents_df = generate_agents()
    customers_df = generate_customers()
    tickets_df = generate_tickets(agents_df, customers_df)
    
    # Create and fit categorizer
    print("Creating and fitting ticket categorizer...")
    categorizer = TicketCategorizer(use_bert=False)  # Using TF-IDF for speed
    
    # Fit model
    categorizer.fit(
        tickets_df,
        text_column='content',
        type_column='type',
        priority_column='priority',
        customer_data=customers_df,
        customer_id_column='customer_id'
    )
    
    # Show model information
    print("\nModel Information:")
    info = categorizer.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Make predictions on a few examples
    sample_tickets = tickets_df.sample(5)
    predictions = categorizer.predict(sample_tickets)
    
    print("\nSample Predictions:")
    for i, (_, ticket) in enumerate(sample_tickets.iterrows()):
        print(f"\nTicket {i+1}: {ticket['content']}")
        print(f"Actual Type: {ticket['type']}")
        print(f"Predicted Type: {predictions.iloc[i]['predicted_type']} (confidence: {predictions.iloc[i]['type_confidence']:.2f})")
        print(f"Actual Priority: {ticket['priority']}")
        print(f"Predicted Priority: {predictions.iloc[i]['predicted_priority']} (confidence: {predictions.iloc[i]['priority_confidence']:.2f})")
    
    # Save model
    os.makedirs('../models/saved', exist_ok=True)
    categorizer.save_model('../models/saved/ticket_categorizer.joblib') 