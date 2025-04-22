#!/usr/bin/env python3
"""
Agent-Ticket Assignment Optimization Model

This module implements optimization algorithms to efficiently assign
customer service tickets to agents based on skills, workload, and priorities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import json
from scipy.optimize import linear_sum_assignment
import pulp
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentAssignmentOptimizer:
    """
    Optimization model for assigning tickets to agents based on 
    skills, workload, priority, and other factors.
    """
    
    def __init__(self, config_path=None):
        """
        Initialize the assignment optimizer.
        
        Parameters:
        -----------
        config_path : str
            Path to configuration file (JSON) with assignment parameters
        """
        # Default configuration
        self.config = {
            'priority_weights': {
                'Low': 1,
                'Medium': 3,
                'High': 6,
                'Critical': 10
            },
            'specialty_bonus': 2.0,  # Multiplier for agent with specialty in ticket type
            'workload_penalty': 0.8,  # Penalty factor for each assigned ticket
            'wait_time_weight': 0.2,  # Weight for ticket wait time
            'max_tickets_per_agent': 15,  # Maximum tickets per agent
            'allow_reassignment': True,  # Whether to allow ticket reassignment
            'reassignment_penalty': 0.7  # Penalty for reassigning tickets
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
                
        # Performance metrics
        self.metrics = {}
        
    def _calculate_agent_ticket_scores(self, agents, tickets, current_assignments=None):
        """
        Calculate the score/cost matrix for assigning tickets to agents.
        
        Parameters:
        -----------
        agents : pandas.DataFrame
            DataFrame with agent information
        tickets : pandas.DataFrame
            DataFrame with ticket information
        current_assignments : dict
            Dictionary of current ticket-agent assignments
            
        Returns:
        --------
        numpy.ndarray
            Score matrix where scores[i,j] is the score for assigning 
            ticket i to agent j. Higher scores are better.
        """
        # Initialize the score matrix
        num_tickets = len(tickets)
        num_agents = len(agents)
        scores = np.zeros((num_tickets, num_agents))
        
        # Get agent workload (number of tickets currently assigned)
        agent_workload = {}
        if current_assignments:
            for ticket_id, agent_id in current_assignments.items():
                agent_workload[agent_id] = agent_workload.get(agent_id, 0) + 1
        
        # Fill in the score matrix
        for i, (_, ticket) in enumerate(tickets.iterrows()):
            ticket_type = ticket['type']
            ticket_priority = ticket['priority']
            ticket_created = ticket['created_at']
            
            # Calculate wait time in hours
            wait_time = (datetime.now() - ticket_created).total_seconds() / 3600
            
            for j, (_, agent) in enumerate(agents.iterrows()):
                agent_id = agent['agent_id']
                
                # Base score from priority
                priority_weight = self.config['priority_weights'].get(
                    ticket_priority, 
                    self.config['priority_weights']['Medium']
                )
                score = priority_weight
                
                # Add wait time factor (older tickets get higher priority)
                score += wait_time * self.config['wait_time_weight']
                
                # Check if agent has specialty in this ticket type
                if ticket_type in agent['specialties']:
                    score *= self.config['specialty_bonus']
                
                # Apply workload penalty
                workload = agent_workload.get(agent_id, 0)
                if workload >= self.config['max_tickets_per_agent']:
                    score = 0  # Agent is at capacity
                else:
                    score *= (1 - workload * self.config['workload_penalty'] / 
                            self.config['max_tickets_per_agent'])
                
                # Apply reassignment penalty if applicable
                if (current_assignments and 
                    ticket['ticket_id'] in current_assignments and 
                    current_assignments[ticket['ticket_id']] != agent_id):
                    score *= self.config['reassignment_penalty']
                
                scores[i, j] = score
                
        return scores
    
    def _hungarian_assignment(self, scores):
        """
        Use the Hungarian algorithm to find the optimal assignment.
        
        Parameters:
        -----------
        scores : numpy.ndarray
            Score matrix where scores[i,j] is the score for assigning 
            ticket i to agent j. Higher scores are better.
            
        Returns:
        --------
        list of tuples
            List of (ticket_idx, agent_idx) assignments
        """
        # For Hungarian algorithm, we need to convert to a cost matrix (minimize cost)
        # So we negate the scores and add the maximum score to ensure all positive
        cost_matrix = np.max(scores) - scores
        
        # Find the optimal assignment (Hungarian algorithm)
        ticket_indices, agent_indices = linear_sum_assignment(cost_matrix)
        
        return list(zip(ticket_indices, agent_indices))
    
    def _linear_programming_assignment(self, scores, tickets, agents):
        """
        Use linear programming to find the optimal assignment with constraints.
        
        Parameters:
        -----------
        scores : numpy.ndarray
            Score matrix where scores[i,j] is the score for assigning 
            ticket i to agent j. Higher scores are better.
        tickets : pandas.DataFrame
            DataFrame with ticket information
        agents : pandas.DataFrame
            DataFrame with agent information
            
        Returns:
        --------
        dict
            Dictionary mapping ticket indices to agent indices
        """
        num_tickets = len(tickets)
        num_agents = len(agents)
        
        # Create the model
        model = pulp.LpProblem("TicketAssignment", pulp.LpMaximize)
        
        # Create variables
        assignments = {}
        for i in range(num_tickets):
            for j in range(num_agents):
                assignments[(i, j)] = pulp.LpVariable(
                    f"assign_t{i}_a{j}", 
                    cat=pulp.LpBinary
                )
        
        # Objective function: maximize total score
        model += pulp.lpSum(
            scores[i, j] * assignments[(i, j)]
            for i in range(num_tickets)
            for j in range(num_agents)
        )
        
        # Constraint: each ticket assigned to at most one agent
        for i in range(num_tickets):
            model += pulp.lpSum(
                assignments[(i, j)] for j in range(num_agents)
            ) <= 1
        
        # Constraint: each agent assigned at most max_tickets tickets
        for j in range(num_agents):
            model += pulp.lpSum(
                assignments[(i, j)] for i in range(num_tickets)
            ) <= self.config['max_tickets_per_agent']
        
        # Solve the problem
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract the solution
        result = {}
        for i in range(num_tickets):
            for j in range(num_agents):
                if pulp.value(assignments[(i, j)]) == 1:
                    result[i] = j
                    
        return result

    def optimize_assignments(self, tickets, agents, current_assignments=None, 
                           method='linear_programming'):
        """
        Optimize ticket-agent assignments.
        
        Parameters:
        -----------
        tickets : pandas.DataFrame
            DataFrame with ticket information
        agents : pandas.DataFrame
            DataFrame with agent information
        current_assignments : dict
            Dictionary of current ticket-agent assignments
        method : str
            Optimization method ('hungarian' or 'linear_programming')
            
        Returns:
        --------
        dict
            Dictionary mapping ticket IDs to agent IDs
        """
        logger.info(f"Optimizing assignments for {len(tickets)} tickets and {len(agents)} agents")
        
        # Calculate the score matrix
        scores = self._calculate_agent_ticket_scores(agents, tickets, current_assignments)
        
        # Get the optimal assignment based on the chosen method
        if method == 'hungarian':
            logger.info("Using Hungarian algorithm for assignment")
            assignments = self._hungarian_assignment(scores)
            
            # Convert to dictionary mapping ticket ID to agent ID
            result = {}
            for ticket_idx, agent_idx in assignments:
                # Only include if the score is positive (valid assignment)
                if scores[ticket_idx, agent_idx] > 0:
                    ticket_id = tickets.iloc[ticket_idx]['ticket_id']
                    agent_id = agents.iloc[agent_idx]['agent_id']
                    result[ticket_id] = agent_id
                    
        elif method == 'linear_programming':
            logger.info("Using Linear Programming for assignment")
            assignment_map = self._linear_programming_assignment(scores, tickets, agents)
            
            # Convert to dictionary mapping ticket ID to agent ID
            result = {}
            for ticket_idx, agent_idx in assignment_map.items():
                # Only include if the score is positive (valid assignment)
                if scores[ticket_idx, agent_idx] > 0:
                    ticket_id = tickets.iloc[ticket_idx]['ticket_id']
                    agent_id = agents.iloc[agent_idx]['agent_id']
                    result[ticket_id] = agent_id
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate performance metrics
        self._calculate_metrics(tickets, agents, result, scores)
        
        logger.info(f"Assignment complete: {len(result)}/{len(tickets)} tickets assigned")
        return result
    
    def _calculate_metrics(self, tickets, agents, assignments, scores):
        """
        Calculate performance metrics for the assignment.
        
        Parameters:
        -----------
        tickets : pandas.DataFrame
            DataFrame with ticket information
        agents : pandas.DataFrame
            DataFrame with agent information
        assignments : dict
            Dictionary mapping ticket IDs to agent IDs
        scores : numpy.ndarray
            Score matrix
        """
        # Create a mapping from agent ID to agent index
        agent_id_to_idx = {agent['agent_id']: i for i, (_, agent) in enumerate(agents.iterrows())}
        
        # Create a mapping from ticket ID to ticket index
        ticket_id_to_idx = {ticket['ticket_id']: i for i, (_, ticket) in enumerate(tickets.iterrows())}
        
        # Calculate average score
        total_score = 0
        for ticket_id, agent_id in assignments.items():
            if ticket_id in ticket_id_to_idx and agent_id in agent_id_to_idx:
                ticket_idx = ticket_id_to_idx[ticket_id]
                agent_idx = agent_id_to_idx[agent_id]
                total_score += scores[ticket_idx, agent_idx]
                
        avg_score = total_score / len(assignments) if assignments else 0
        
        # Calculate agent utilization
        agent_tickets = {}
        for agent_id in agents['agent_id']:
            agent_tickets[agent_id] = 0
            
        for ticket_id, agent_id in assignments.items():
            agent_tickets[agent_id] = agent_tickets.get(agent_id, 0) + 1
            
        utilization = [count / self.config['max_tickets_per_agent'] 
                      for count in agent_tickets.values()]
        avg_utilization = np.mean(utilization)
        utilization_std = np.std(utilization)
        
        # Calculate priority distribution
        priorities = {}
        for _, ticket in tickets.iterrows():
            if ticket['ticket_id'] in assignments:
                priority = ticket['priority']
                priorities[priority] = priorities.get(priority, 0) + 1
                
        # Calculate specialty match rate
        specialty_matches = 0
        for ticket_id, agent_id in assignments.items():
            ticket_info = tickets[tickets['ticket_id'] == ticket_id]
            agent_info = agents[agents['agent_id'] == agent_id]
            
            if not ticket_info.empty and not agent_info.empty:
                ticket_type = ticket_info.iloc[0]['type']
                agent_specialties = agent_info.iloc[0]['specialties']
                
                if ticket_type in agent_specialties:
                    specialty_matches += 1
                    
        specialty_match_rate = specialty_matches / len(assignments) if assignments else 0
        
        # Store metrics
        self.metrics = {
            'total_tickets': len(tickets),
            'assigned_tickets': len(assignments),
            'assignment_rate': len(assignments) / len(tickets) if tickets.shape[0] > 0 else 0,
            'avg_score': avg_score,
            'avg_utilization': avg_utilization,
            'utilization_std': utilization_std,
            'specialty_match_rate': specialty_match_rate,
            'priority_distribution': priorities
        }
        
        logger.info(f"Assignment metrics: {self.metrics}")
    
    def get_metrics(self):
        """
        Get the performance metrics for the last assignment.
        
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        return self.metrics
    
    def save_config(self, path):
        """
        Save the configuration to a file.
        
        Parameters:
        -----------
        path : str
            Path to save the configuration
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info(f"Configuration saved to {path}")
    
    def load_config(self, path):
        """
        Load the configuration from a file.
        
        Parameters:
        -----------
        path : str
            Path to load the configuration from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r') as f:
            self.config = json.load(f)
            
        logger.info(f"Configuration loaded from {path}")
        
    def get_agent_workload(self, assignments):
        """
        Calculate the current workload for each agent.
        
        Parameters:
        -----------
        assignments : dict
            Dictionary mapping ticket IDs to agent IDs
            
        Returns:
        --------
        dict
            Dictionary mapping agent IDs to number of assigned tickets
        """
        workload = {}
        for ticket_id, agent_id in assignments.items():
            workload[agent_id] = workload.get(agent_id, 0) + 1
            
        return workload
    
    def optimize_agent_scheduling(self, tickets_forecast, available_agents, 
                                shift_duration=8, window_days=7):
        """
        Optimize agent scheduling based on ticket volume forecast.
        
        Parameters:
        -----------
        tickets_forecast : pandas.DataFrame
            DataFrame with forecasted ticket volume
        available_agents : pandas.DataFrame
            DataFrame with available agents and their attributes
        shift_duration : int
            Duration of each shift in hours
        window_days : int
            Number of days to schedule ahead
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with agent schedule
        """
        logger.info(f"Optimizing agent scheduling for {window_days} days ahead")
        
        # Get the number of time periods
        num_periods = window_days * 24 // shift_duration
        num_agents = len(available_agents)
        
        # Create a linear programming model
        model = pulp.LpProblem("AgentScheduling", pulp.LpMinimize)
        
        # Create variables
        # x[i, t] = 1 if agent i works in time period t
        x = {}
        for i in range(num_agents):
            for t in range(num_periods):
                x[i, t] = pulp.LpVariable(f"agent_{i}_period_{t}", cat=pulp.LpBinary)
        
        # Calculate the number of agents needed per period based on forecast
        periods = []
        agents_needed = []
        for day in range(window_days):
            for shift in range(24 // shift_duration):
                period_start = day * 24 + shift * shift_duration
                period_end = period_start + shift_duration
                
                # Filter forecast for this period
                period_forecast = tickets_forecast[
                    (tickets_forecast.index.hour >= period_start % 24) &
                    (tickets_forecast.index.hour < period_end % 24) &
                    (tickets_forecast.index.dayofweek == day % 7)
                ]
                
                # Calculate agents needed (assuming 2 tickets per hour per agent)
                if not period_forecast.empty:
                    tickets_per_hour = period_forecast.mean()
                    needed = np.ceil(tickets_per_hour / 2)
                else:
                    needed = 1  # Default - at least one agent
                
                periods.append((period_start, period_end))
                agents_needed.append(needed)
        
        # Objective: minimize the total number of agent-hours
        model += pulp.lpSum(x[i, t] for i in range(num_agents) for t in range(num_periods))
        
        # Constraint: each time period needs enough agents
        for t in range(num_periods):
            model += pulp.lpSum(x[i, t] for i in range(num_agents)) >= agents_needed[t]
        
        # Constraint: an agent can't work more than 5 days a week
        periods_per_day = 24 // shift_duration
        for i in range(num_agents):
            for day in range(window_days // 7):  # For each week
                weekly_periods = range(day * 7 * periods_per_day, (day + 1) * 7 * periods_per_day)
                model += pulp.lpSum(x[i, t] for t in weekly_periods) <= 5 * periods_per_day
        
        # Constraint: an agent needs at least 12 hours between shifts
        min_rest_periods = 12 // shift_duration
        for i in range(num_agents):
            for t in range(num_periods - min_rest_periods):
                # If agent works at time t, they can't work for the next min_rest_periods
                consecutive_shifts = range(t + 1, min(t + min_rest_periods + 1, num_periods))
                model += x[i, t] + pulp.lpSum(x[i, s] for s in consecutive_shifts) <= min_rest_periods
        
        # Solve the problem
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract the solution
        schedule = []
        for i in range(num_agents):
            agent_id = available_agents.iloc[i]['agent_id']
            agent_name = available_agents.iloc[i]['name']
            
            for t in range(num_periods):
                if pulp.value(x[i, t]) == 1:
                    period_start, period_end = periods[t]
                    day = period_start // 24
                    shift_start = period_start % 24
                    shift_end = period_end % 24
                    
                    schedule.append({
                        'agent_id': agent_id,
                        'agent_name': agent_name,
                        'day': day,
                        'shift_start': shift_start,
                        'shift_end': shift_end,
                        'date': (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
                    })
        
        schedule_df = pd.DataFrame(schedule)
        
        logger.info(f"Scheduling complete: {len(schedule)} shifts assigned")
        return schedule_df


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
    
    # Create optimizer
    print("Creating assignment optimizer...")
    optimizer = AgentAssignmentOptimizer()
    
    # Filter for open tickets
    open_tickets = tickets_df[tickets_df['status'] == 'Open']
    print(f"Found {len(open_tickets)} open tickets")
    
    # Get available agents
    available_agents = agents_df.sample(min(len(agents_df), 10))  # Simulate 10 available agents
    print(f"Using {len(available_agents)} available agents")
    
    # Optimize assignments
    assignments = optimizer.optimize_assignments(
        open_tickets,
        available_agents,
        method='linear_programming'
    )
    
    # Print results
    print("\nAssignment Results:")
    print(f"Total tickets: {len(open_tickets)}")
    print(f"Assigned tickets: {len(assignments)}")
    print(f"Assignment rate: {len(assignments) / len(open_tickets) * 100:.1f}%")
    
    # Print agent workload
    workload = optimizer.get_agent_workload(assignments)
    print("\nAgent Workload:")
    for agent_id, count in workload.items():
        agent_name = agents_df[agents_df['agent_id'] == agent_id]['name'].values[0]
        print(f"{agent_name} ({agent_id}): {count} tickets")
    
    # Print metrics
    metrics = optimizer.get_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if key != 'priority_distribution':
            print(f"{key}: {value}")
            
    print("\nPriority Distribution:")
    for priority, count in metrics.get('priority_distribution', {}).items():
        print(f"{priority}: {count} tickets")
    
    # Save configuration
    os.makedirs('../models/config', exist_ok=True)
    optimizer.save_config('../models/config/assignment_config.json') 