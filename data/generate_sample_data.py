#!/usr/bin/env python3
"""
Sample data generator for customer service ticket data.
Creates realistic datasets for modeling and dashboard development.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import random
from faker import Faker

# Initialize faker
fake = Faker()

# Constants
NUM_TICKETS = 5000
NUM_AGENTS = 50
NUM_CUSTOMERS = 2000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)
TICKET_TYPES = ['Account Issue', 'Technical Support', 'Billing Question', 
                'Feature Request', 'Bug Report', 'Password Reset',
                'Account Verification', 'Transfer Inquiry', 'App Issue',
                'Security Concern']
TICKET_CHANNELS = ['Email', 'Phone', 'Chat', 'Social Media', 'Mobile App']
PRIORITIES = ['Low', 'Medium', 'High', 'Critical']
STATUS = ['Open', 'In Progress', 'Resolved', 'Closed', 'Reopened']
TEAMS = ['General Support', 'Technical', 'Account Management', 'Billing', 'Security']

# Seasonal and time-of-day patterns for realistic ticket volume simulation
def seasonal_pattern(day_of_year):
    """Return a seasonal multiplier (higher in Q1 and Q4)"""
    # Tax season (Q1) and holidays (Q4) have higher volumes
    if day_of_year < 90 or day_of_year > 330:  
        return 1.2
    elif 150 < day_of_year < 240:  # Summer months
        return 0.8
    else:
        return 1.0

def time_of_day_pattern(hour):
    """Return a time-of-day multiplier (higher during business hours)"""
    if 9 <= hour < 17:  # Business hours
        return 1.5
    elif 7 <= hour < 9 or 17 <= hour < 20:  # Morning and evening
        return 1.2
    elif 0 <= hour < 5:  # Very early morning
        return 0.3
    else:
        return 0.7

def generate_agents():
    """Generate agent data"""
    agents = []
    for i in range(NUM_AGENTS):
        skill_level = np.random.choice(['Junior', 'Mid', 'Senior'], p=[0.3, 0.5, 0.2])
        team = np.random.choice(TEAMS)
        
        # Experience level affects resolution time
        if skill_level == 'Junior':
            avg_resolution_time = np.random.uniform(50, 90)
        elif skill_level == 'Mid':
            avg_resolution_time = np.random.uniform(30, 60)
        else:
            avg_resolution_time = np.random.uniform(15, 45)
            
        # Create specialties - agents are better at certain ticket types
        specialties = np.random.choice(TICKET_TYPES, 
                                       size=np.random.randint(1, 4), 
                                       replace=False).tolist()
        
        agents.append({
            'agent_id': f'AG{i+1:03d}',
            'name': fake.name(),
            'team': team,
            'skill_level': skill_level,
            'specialties': specialties,
            'avg_resolution_time': avg_resolution_time,
            'join_date': fake.date_between(start_date=START_DATE - timedelta(days=365*3), 
                                         end_date=END_DATE)
        })
    
    return pd.DataFrame(agents)

def generate_customers():
    """Generate customer data"""
    customers = []
    for i in range(NUM_CUSTOMERS):
        signup_date = fake.date_between(start_date=START_DATE - timedelta(days=365*5), 
                                       end_date=END_DATE)
        
        # Some customers have more tickets than others
        ticket_frequency = np.random.choice(['Low', 'Medium', 'High'], 
                                          p=[0.7, 0.2, 0.1])
        
        customers.append({
            'customer_id': f'CU{i+1:04d}',
            'name': fake.name(),
            'email': fake.email(),
            'signup_date': signup_date,
            'account_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], 
                                           p=[0.6, 0.3, 0.1]),
            'ticket_frequency': ticket_frequency,
            'country': fake.country(),
            'language': np.random.choice(['English', 'French', 'Spanish', 'Chinese'], 
                                       p=[0.7, 0.1, 0.1, 0.1])
        })
    
    return pd.DataFrame(customers)

def generate_tickets(agents_df, customers_df):
    """Generate ticket data with realistic patterns"""
    tickets = []
    
    # More frequent customers create more tickets
    customer_weights = {
        'Low': 1,
        'Medium': 3,
        'High': 8
    }
    
    # Create weighted customer selection based on ticket frequency
    customer_probabilities = np.array([customer_weights[freq] for freq in customers_df['ticket_frequency']])
    customer_probabilities = customer_probabilities / customer_probabilities.sum()
    
    for i in range(NUM_TICKETS):
        # Choose a customer based on their ticket frequency
        customer = customers_df.iloc[np.random.choice(len(customers_df), p=customer_probabilities)]
        
        # Generate a random timestamp with weekday, time-of-day patterns
        random_days = np.random.randint(0, (END_DATE - START_DATE).days)
        random_hour = np.random.randint(0, 24)
        
        # Apply seasonal and time-of-day patterns
        while True:
            # Keep trying until we find a timestamp that passes our probability filter
            random_days = np.random.randint(0, (END_DATE - START_DATE).days)
            random_hour = np.random.randint(0, 24)
            timestamp = START_DATE + timedelta(days=random_days, hours=random_hour)
            
            day_of_year = timestamp.timetuple().tm_yday
            seasonal_factor = seasonal_pattern(day_of_year)
            time_factor = time_of_day_pattern(random_hour)
            
            # Apply day of week pattern (weekdays have more tickets)
            weekday_factor = 1.2 if timestamp.weekday() < 5 else 0.6
            
            # Combined probability
            combined_probability = seasonal_factor * time_factor * weekday_factor
            
            if np.random.random() < combined_probability:
                break
        
        # Add some minutes and seconds for realism
        timestamp += timedelta(minutes=np.random.randint(0, 60), 
                              seconds=np.random.randint(0, 60))
        
        # Generate ticket type with some customers having specific problem patterns
        account_type_factor = {'Basic': 0.1, 'Premium': 0, 'Enterprise': -0.1}
        
        # Technical issues more common for basic accounts
        tech_issue_prob = 0.3 + account_type_factor[customer['account_type']]
        
        if np.random.random() < tech_issue_prob:
            ticket_type = np.random.choice(['Technical Support', 'App Issue', 'Bug Report'])
        else:
            ticket_type = np.random.choice(TICKET_TYPES)
        
        # Generate content based on ticket type
        if ticket_type == 'Account Issue':
            content = np.random.choice([
                "I can't access my account",
                "Need help updating my account information",
                "My account shows incorrect information",
                "Getting an error when trying to log in"
            ])
        elif ticket_type == 'Technical Support':
            content = np.random.choice([
                "App is crashing when I try to view my portfolio",
                "Website is not loading properly",
                "Getting error code 500 when accessing my account",
                "Can't complete my transaction due to technical error"
            ])
        elif ticket_type == 'Billing Question':
            content = np.random.choice([
                "Why was I charged this fee?",
                "Need explanation about recent transaction",
                "Is there a fee for transferring funds?",
                "How often are management fees charged?"
            ])
        elif ticket_type == 'Password Reset':
            content = "I need to reset my password"
        elif ticket_type == 'Transfer Inquiry':
            content = np.random.choice([
                "How long does a transfer take to process?",
                "My transfer has been pending for several days",
                "Can I cancel a transfer in progress?",
                "Need to update my linked bank account"
            ])
        else:
            content = f"Question regarding {ticket_type.lower()}"
            
        # Assign priority
        if 'Critical' in content or 'urgent' in content.lower() or ticket_type in ['Security Concern', 'Bug Report']:
            priority = np.random.choice(['High', 'Critical'], p=[0.7, 0.3])
        else:
            priority = np.random.choice(PRIORITIES, p=[0.4, 0.4, 0.15, 0.05])
            
        # Calculate response and resolution times
        response_time = None
        resolution_time = None
        resolved_at = None
        status = np.random.choice(STATUS, p=[0.1, 0.2, 0.4, 0.25, 0.05])
        
        # Assign agent based on ticket type and specialties
        eligible_agents = agents_df[agents_df['specialties'].apply(lambda x: ticket_type in x)]
        if len(eligible_agents) > 0:
            agent = eligible_agents.sample(1).iloc[0]
        else:
            agent = agents_df.sample(1).iloc[0]
            
        # Calculate response time (in minutes)
        if status != 'Open':
            # Response time affected by priority
            priority_factor = {'Low': 120, 'Medium': 60, 'High': 30, 'Critical': 15}
            mean_response_time = priority_factor[priority]
            response_time = max(5, np.random.normal(mean_response_time, mean_response_time/3))
            
            # Calculate resolution time and timestamp if applicable
            if status in ['Resolved', 'Closed']:
                base_resolution_time = agent['avg_resolution_time']
                
                # Adjust for agent specialty
                specialty_factor = 0.7 if ticket_type in agent['specialties'] else 1.3
                
                # Adjust for priority
                priority_resolution_factor = {'Low': 1.0, 'Medium': 1.2, 'High': 1.5, 'Critical': 2.0}
                
                # Calculate final resolution time
                resolution_time = max(10, base_resolution_time * specialty_factor * 
                                   priority_resolution_factor[priority] * 
                                   np.random.normal(1, 0.2))
                
                resolved_at = timestamp + timedelta(minutes=response_time + resolution_time)
                
        # Create the ticket
        ticket = {
            'ticket_id': f'TK{i+1:05d}',
            'customer_id': customer['customer_id'],
            'agent_id': agent['agent_id'] if status != 'Open' else None,
            'created_at': timestamp,
            'type': ticket_type,
            'channel': np.random.choice(TICKET_CHANNELS),
            'priority': priority,
            'status': status,
            'content': content,
            'response_time_min': response_time,
            'resolution_time_min': resolution_time,
            'resolved_at': resolved_at,
            'satisfaction_score': np.random.choice([1, 2, 3, 4, 5, None], 
                                                 p=[0.05, 0.05, 0.1, 0.2, 0.4, 0.2]) 
                                 if status in ['Resolved', 'Closed'] else None
        }
        
        tickets.append(ticket)
    
    return pd.DataFrame(tickets)

def generate_agent_shifts():
    """Generate agent shift schedule data"""
    shifts = []
    # Generate 3 months of shifts
    start_date = datetime(2023, 10, 1)
    end_date = datetime(2023, 12, 31)
    
    # Create shift types
    shift_types = {
        'Morning': (8, 16),  # 8 AM to 4 PM
        'Afternoon': (12, 20),  # 12 PM to 8 PM
        'Evening': (16, 24),  # 4 PM to 12 AM
        'Night': (0, 8),  # 12 AM to 8 AM
    }
    
    # Each agent gets assigned shifts
    for agent_id in range(1, NUM_AGENTS + 1):
        agent_id = f'AG{agent_id:03d}'
        
        # Assign a general shift pattern to this agent
        primary_shift = np.random.choice(list(shift_types.keys()), p=[0.4, 0.3, 0.2, 0.1])
        
        current_date = start_date
        while current_date <= end_date:
            # Skip some days for time off (weekends more likely)
            if current_date.weekday() >= 5:  # Weekend
                if np.random.random() < 0.7:  # 70% chance of day off on weekend
                    current_date += timedelta(days=1)
                    continue
            else:  # Weekday
                if np.random.random() < 0.1:  # 10% chance of day off on weekday
                    current_date += timedelta(days=1)
                    continue
            
            # Determine shift for this day (occasionally agent works different shift)
            if np.random.random() < 0.8:
                shift = primary_shift
            else:
                shift = np.random.choice([s for s in shift_types.keys() if s != primary_shift])
            
            start_hour, end_hour = shift_types[shift]
            
            shift_start = datetime(current_date.year, current_date.month, current_date.day, 
                                 start_hour, 0, 0)
            shift_end = datetime(current_date.year, current_date.month, current_date.day, 
                               end_hour, 0, 0)
            
            # If shift crosses midnight
            if start_hour > end_hour:
                shift_end = shift_end + timedelta(days=1)
            
            shifts.append({
                'agent_id': agent_id,
                'shift_start': shift_start,
                'shift_end': shift_end,
                'shift_type': shift
            })
            
            current_date += timedelta(days=1)
    
    return pd.DataFrame(shifts)

def generate_all_data():
    """Generate all datasets and save to files"""
    print("Generating agent data...")
    agents_df = generate_agents()
    
    print("Generating customer data...")
    customers_df = generate_customers()
    
    print("Generating ticket data...")
    tickets_df = generate_tickets(agents_df, customers_df)
    
    print("Generating agent shift data...")
    shifts_df = generate_agent_shifts()
    
    # Create output directory if it doesn't exist
    os.makedirs('raw', exist_ok=True)
    
    # Save to CSV
    agents_df.to_csv('raw/agents.csv', index=False)
    customers_df.to_csv('raw/customers.csv', index=False)
    tickets_df.to_csv('raw/tickets.csv', index=False)
    shifts_df.to_csv('raw/agent_shifts.csv', index=False)
    
    print("Data generation complete! Files saved to 'raw' directory.")

if __name__ == "__main__":
    generate_all_data() 