import pandas as pd
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, LpStatus, value

# Load dataset
df = pd.read_csv('hybrid_manufacturing.csv')
df.head()

# Filter only jobs with valid numerical values
df = df[df['Job_Status'] != 'Failed'].copy()
df.dropna(subset=['Material_Used', 'Processing_Time', 'Energy_Consumption'], inplace=True)

# Create binary decision variables for each job: 1 if we select/schedule it, 0 otherwise
job_vars = {
    row['Job_ID']: LpVariable(name=f"select_{row['Job_ID']}", cat=LpBinary)
    for _, row in df.iterrows()
}

# Define resource constraints
max_material = 50
max_time = 600
max_energy = 150      

# Initialize the LP model
model = LpProblem("Job_Scheduling_Optimization", LpMaximize)

# Objective: Maximize the number of selected jobs
model += lpSum([job_vars[job_id] for job_id in job_vars]), "Total_Completed_Jobs"

# Add constraints
model += lpSum([row['Material_Used'] * job_vars[row['Job_ID']] for _, row in df.iterrows()]) <= max_material, "Material_Limit"
model += lpSum([row['Processing_Time'] * job_vars[row['Job_ID']] for _, row in df.iterrows()]) <= max_time, "Time_Limit"
model += lpSum([row['Energy_Consumption'] * job_vars[row['Job_ID']] for _, row in df.iterrows()]) <= max_energy, "Energy_Limit"

# Solve the model
status = model.solve()
print(f"Status: {LpStatus[model.status]}")

# Output the results
print("\nSelected Jobs:")
selected_jobs = []
for job_id, var in job_vars.items():
    if var.value() == 1:
        selected_jobs.append(job_id)
        print(job_id)

print(f"\nTotal Jobs Scheduled: {len(selected_jobs)}")
print(f"Total Objective Value: {value(model.objective)}")
