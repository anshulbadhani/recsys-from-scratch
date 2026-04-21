import pandas as pd
import pyarrow.json as paj
from config import DATA_DIR

table = paj.read_json(DATA_DIR / "reviews_Software.jsonl.gz")
df = table.to_pandas()

# 0. Deduplicate FIRST — some users reviewed the same item twice
#    Keep the most recent review per user-item pair.
#    Must happen before filtering so interaction counts are accurate.
df = df.sort_values(['user_id', 'parent_asin', 'timestamp'])
df = df.drop_duplicates(subset=['user_id', 'parent_asin'], keep='last')
print(f"After dedup:   {len(df):,} interactions")

# 1. Sort by timestamp — critical for leave-one-out correctness
df = df.sort_values(['user_id', 'timestamp', 'parent_asin'])  # parent_asin breaks ties deterministically

# 2. Keep only users with 5+ interactions
user_counts = df.groupby('user_id').size()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]

# 3. Keep items that appear at least 5 times
item_counts = df.groupby('parent_asin').size()
df = df[df['parent_asin'].isin(item_counts[item_counts >= 5].index)]

# 4. Re-filter users after item filtering
user_counts = df.groupby('user_id').size()
df = df[df['user_id'].isin(user_counts[user_counts >= 5].index)]

print(f"Users:         {df['user_id'].nunique():,}")
print(f"Items:         {df['parent_asin'].nunique():,}")
print(f"Interactions:  {len(df):,}")

# 5. Leave-one-out split
df['rank']     = df.groupby('user_id')['timestamp'].rank(method='first', ascending=True)
df['max_rank'] = df.groupby('user_id')['rank'].transform('max')

train_df = df[df['rank'] <  df['max_rank']]
test_df  = df[df['rank'] == df['max_rank']]

print(f"Train interactions: {len(train_df):,}")
print(f"Test  users:        {len(test_df):,}")

# 6. Save
train_df[['user_id', 'parent_asin', 'rating', 'timestamp']].to_csv(DATA_DIR / 'train.csv', index=False)
test_df[['user_id',  'parent_asin']].to_csv(DATA_DIR / 'test.csv',  index=False)