import pandas as pd

# Your original data frame
data = {
    'location_id': [66, 67, 68, 69],
    'pop_ratio': [0.001919, 0.720001, 0.255514, 0.022566]
}

pop_df = pd.DataFrame(data)

print("Original data frame:")
print(pop_df)
print()

# Transform to have pop_ratio as rows and location_id as columns
transformed_df = pop_df.set_index('location_id').T

print("Transformed data frame (pop_ratio as rows, location_id as columns):")
print(transformed_df)
print()

# Alternative method using pivot_table
transformed_df_alt = pop_df.pivot_table(
    index=None, 
    columns='location_id', 
    values='pop_ratio'
)

print("Alternative transformation using pivot_table:")
print(transformed_df_alt) 