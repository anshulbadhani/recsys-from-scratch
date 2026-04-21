import pandas as pd

# Load both files
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")

user = "AHZZOSNJHLAQC66I6D35WSFMJURA"
asin = "B01BGYEOYG"

# Show all their interactions from train
user_train = train_df[train_df['user_id'] == user].sort_values('timestamp')
print("Train history:")
print(user_train)

print("\nTest entry:")
print(test_df[test_df['user_id'] == user])