import pandas as pd
from datetime import datetime, timedelta

# Sample synthetic data for parts
data = {
    'part_id': ['Part1', 'Part2', 'Part3', 'Part4'],
    'project': ['ProjectA', 'ProjectB', 'ProjectA', 'ProjectC'],
    'project_importance': [3, 1, 3, 2],
    'delivery_date': [datetime(2024, 11, 15), datetime(2024, 11, 12), datetime(2024, 11, 20), datetime(2024, 11, 18)],
    'num_parts_in_plate': [1, 3, 2, 1],
    'material_type': ['ABS', 'PLA', 'ABS', 'Nylon'],
    'build_height': [100, 150, 120, 90],
    'print_time': [5, 3, 4, 2]  # Estimated print time in hours
}

# Sample synthetic data for printers
printer_data = {
    'printer_id': ['Printer1', 'Printer2'],
    'max_build_height': [150, 120],
    'assigned_material': ['ABS', 'PLA'],
    'available_from': [datetime.now(), datetime.now()]  # Printer availability
}

# Create DataFrames
parts = pd.DataFrame(data)
printers = pd.DataFrame(printer_data)

# Function to assign priority based on project importance, delivery date, and number of parts in plate
def assign_priority(parts_df):
    # Calculate days until delivery and normalize
    parts_df['days_until_delivery'] = (parts_df['delivery_date'] - datetime.now()).dt.days
    parts_df['delivery_norm'] = 1 - (parts_df['days_until_delivery'] / parts_df['days_until_delivery'].max())

    # Normalize project importance and number of parts in plate
    parts_df['importance_norm'] = parts_df['project_importance'] / parts_df['project_importance'].max()
    parts_df['plate_norm'] = parts_df['num_parts_in_plate'] / parts_df['num_parts_in_plate'].max()

    # Calculate priority score
    parts_df['priority_score'] = (
        0.5 * parts_df['importance_norm'] +  
        0.3 * parts_df['delivery_norm'] +    
        0.2 * parts_df['plate_norm']
    )

    # Sort parts by priority score in descending order
    parts_df = parts_df.sort_values(by='priority_score', ascending=False).reset_index(drop=True)
    return parts_df

# Function to schedule jobs on available printers based on priority
def schedule_jobs(parts_df, printers_df):
    scheduled_jobs = []

    for _, part in parts_df.iterrows():
        # Check for a compatible printer (based on build height and material)
        for p_index, printer in printers_df.iterrows():
            if (
                part['build_height'] <= printer['max_build_height'] and
                part['material_type'] == printer['assigned_material'] and
                printer['available_from'] <= datetime.now()
            ):
                # Schedule the job
                end_time = datetime.now() + timedelta(hours=part['print_time'])
                printers_df.at[p_index, 'available_from'] = end_time  # Update printer availability
                scheduled_jobs.append({
                    'part_id': part['part_id'],
                    'printer_id': printer['printer_id'],
                    'start_time': datetime.now(),
                    'end_time': end_time,
                    'priority_score': part['priority_score']
                })
                print(f"Assigned {part['part_id']} to {printer['printer_id']} from {datetime.now()} to {end_time}")
                break
        else:
            print(f"No compatible printer available for {part['part_id']} at this time.")

    return scheduled_jobs

# Main workflow
# Step 1: Assign priority to each part
prioritized_parts = assign_priority(parts)
print("Prioritized parts:\n", prioritized_parts[['part_id', 'priority_score']])

# Step 2: Schedule parts based on assigned priority and printer availability
scheduled_jobs = schedule_jobs(prioritized_parts, printers)
scheduled_jobs_df = pd.DataFrame(scheduled_jobs)

# Display final schedule
print("\nFinal Job Schedule:\n", scheduled_jobs_df[['part_id', 'printer_id', 'start_time', 'end_time', 'priority_score']])
