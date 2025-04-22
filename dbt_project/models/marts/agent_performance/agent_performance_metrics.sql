{{
    config(
        materialized='table',
        schema='marts',
        tags=['agent', 'performance', 'daily']
    )
}}

WITH agent_base AS (
    SELECT
        agent_id,
        name AS agent_name,
        team,
        skill_level,
        join_date
    FROM {{ ref('stg_agents') }}
),

-- Get ticket assignment and resolution data
ticket_resolutions AS (
    SELECT
        t.ticket_id,
        t.agent_id,
        t.created_at,
        t.resolved_at,
        t.type AS ticket_type,
        t.priority,
        t.status,
        t.response_time_min,
        t.resolution_time_min,
        t.satisfaction_score,
        
        -- Calculate SLA metrics
        CASE 
            WHEN t.priority = 'Critical' AND t.response_time_min <= {{ var('sla_thresholds')['critical'] }} THEN true
            WHEN t.priority = 'High' AND t.response_time_min <= {{ var('sla_thresholds')['high'] }} THEN true
            WHEN t.priority = 'Medium' AND t.response_time_min <= {{ var('sla_thresholds')['medium'] }} THEN true
            WHEN t.priority = 'Low' AND t.response_time_min <= {{ var('sla_thresholds')['low'] }} THEN true
            ELSE false
        END AS within_response_sla,
        
        -- First contact resolution (no reopens)
        CASE WHEN t.status != 'Reopened' THEN true ELSE false END AS first_contact_resolution
    FROM {{ ref('stg_tickets') }} t
    WHERE t.agent_id IS NOT NULL
),

-- Daily metrics per agent
daily_metrics AS (
    SELECT
        agent_id,
        DATE_TRUNC('day', created_at) AS date,
        
        -- Volume metrics
        COUNT(ticket_id) AS tickets_assigned,
        COUNT(CASE WHEN status IN {{ var('resolved_statuses') }} THEN ticket_id END) AS tickets_resolved,
        
        -- Resolution metrics
        AVG(resolution_time_min) AS avg_resolution_time,
        AVG(response_time_min) AS avg_response_time,
        
        -- SLA metrics
        SUM(CASE WHEN within_response_sla THEN 1 ELSE 0 END) AS tickets_within_sla,
        
        -- FCR metrics
        SUM(CASE WHEN first_contact_resolution THEN 1 ELSE 0 END) AS first_contact_resolutions,
        
        -- CSAT metrics
        AVG(satisfaction_score) AS avg_satisfaction,
        COUNT(satisfaction_score) AS satisfaction_responses,
        
        -- Ticket mix
        SUM(CASE WHEN ticket_type = 'Technical Support' THEN 1 ELSE 0 END) AS technical_tickets,
        SUM(CASE WHEN ticket_type = 'Account Issue' THEN 1 ELSE 0 END) AS account_tickets,
        SUM(CASE WHEN ticket_type = 'Billing Question' THEN 1 ELSE 0 END) AS billing_tickets,
        SUM(CASE WHEN priority = 'Critical' THEN 1 ELSE 0 END) AS critical_tickets,
        SUM(CASE WHEN priority = 'High' THEN 1 ELSE 0 END) AS high_priority_tickets
    FROM ticket_resolutions
    GROUP BY 1, 2
),

-- Add agent shifts data
shift_metrics AS (
    SELECT
        agent_id,
        DATE_TRUNC('day', shift_start) AS date,
        COUNT(*) AS shifts_worked,
        SUM(EXTRACT(EPOCH FROM (shift_end - shift_start)) / 3600) AS hours_worked
    FROM {{ ref('stg_agent_shifts') }}
    GROUP BY 1, 2
),

-- Join all data together
final AS (
    SELECT
        a.agent_id,
        a.agent_name,
        a.team,
        a.skill_level,
        dm.date,
        
        -- Shift info
        COALESCE(sm.shifts_worked, 0) AS shifts_worked,
        COALESCE(sm.hours_worked, 0) AS hours_worked,
        
        -- Volume metrics
        COALESCE(dm.tickets_assigned, 0) AS tickets_assigned,
        COALESCE(dm.tickets_resolved, 0) AS tickets_resolved,
        
        -- Calculate ticket handling rate (tickets per hour)
        CASE 
            WHEN COALESCE(sm.hours_worked, 0) > 0 
            THEN COALESCE(dm.tickets_resolved, 0) / sm.hours_worked 
            ELSE 0 
        END AS tickets_per_hour,
        
        -- Performance metrics
        COALESCE(dm.avg_resolution_time, 0) AS avg_resolution_time_min,
        COALESCE(dm.avg_response_time, 0) AS avg_response_time_min,
        
        -- SLA compliance
        CASE 
            WHEN COALESCE(dm.tickets_assigned, 0) > 0 
            THEN COALESCE(dm.tickets_within_sla, 0) / dm.tickets_assigned 
            ELSE 0 
        END AS sla_compliance_rate,
        
        -- First contact resolution rate
        CASE 
            WHEN COALESCE(dm.tickets_resolved, 0) > 0 
            THEN COALESCE(dm.first_contact_resolutions, 0) / dm.tickets_resolved 
            ELSE 0 
        END AS first_contact_resolution_rate,
        
        -- CSAT metrics
        COALESCE(dm.avg_satisfaction, 0) AS avg_satisfaction_score,
        COALESCE(dm.satisfaction_responses, 0) AS satisfaction_responses,
        
        -- Ticket mix breakdown (percentages)
        CASE 
            WHEN COALESCE(dm.tickets_assigned, 0) > 0 
            THEN COALESCE(dm.technical_tickets, 0) / dm.tickets_assigned 
            ELSE 0 
        END AS technical_ticket_percentage,
        CASE 
            WHEN COALESCE(dm.tickets_assigned, 0) > 0 
            THEN COALESCE(dm.account_tickets, 0) / dm.tickets_assigned 
            ELSE 0 
        END AS account_ticket_percentage,
        CASE 
            WHEN COALESCE(dm.tickets_assigned, 0) > 0 
            THEN COALESCE(dm.billing_tickets, 0) / dm.tickets_assigned 
            ELSE 0 
        END AS billing_ticket_percentage,
        CASE 
            WHEN COALESCE(dm.tickets_assigned, 0) > 0 
            THEN COALESCE(dm.critical_tickets, 0) / dm.tickets_assigned 
            ELSE 0 
        END AS critical_ticket_percentage,
        CASE 
            WHEN COALESCE(dm.tickets_assigned, 0) > 0 
            THEN COALESCE(dm.high_priority_tickets, 0) / dm.tickets_assigned 
            ELSE 0 
        END AS high_priority_ticket_percentage,
        
        -- Raw counts for ticket mix
        COALESCE(dm.technical_tickets, 0) AS technical_tickets,
        COALESCE(dm.account_tickets, 0) AS account_tickets,
        COALESCE(dm.billing_tickets, 0) AS billing_tickets,
        COALESCE(dm.critical_tickets, 0) AS critical_tickets,
        COALESCE(dm.high_priority_tickets, 0) AS high_priority_tickets,
        
        -- Add date hierarchy for easy filtering/grouping
        EXTRACT(YEAR FROM dm.date) AS year,
        EXTRACT(MONTH FROM dm.date) AS month,
        EXTRACT(QUARTER FROM dm.date) AS quarter,
        EXTRACT(DOW FROM dm.date) AS day_of_week,
        CASE WHEN EXTRACT(DOW FROM dm.date) IN (0, 6) THEN true ELSE false END AS is_weekend
    FROM agent_base a
    LEFT JOIN daily_metrics dm 
        ON a.agent_id = dm.agent_id
    LEFT JOIN shift_metrics sm 
        ON a.agent_id = sm.agent_id 
        AND dm.date = sm.date
    WHERE dm.date >= '{{ var("start_date") }}'
)

SELECT * FROM final
ORDER BY date DESC, agent_id 