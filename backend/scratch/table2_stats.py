import pandas as pd, json

df = pd.read_parquet(r'C:\orbital\backend\data\processed\pairs.parquet')

split_ts = df['created_ts'].quantile(0.8)
train = df[df['created_ts'] <= split_ts]
test  = df[df['created_ts'] >  split_ts]

print("=== TRAIN SPLIT ===")
print("  Rows:", len(train))
print("  Positive (label=1):", int(train['label'].sum()))
print("  Negative (label=0):", int((train['label']==0).sum()))
rate_tr = train['label'].mean()*100
print("  Positive rate: {:.2f}%".format(rate_tr))
print("  Date range:", train['created_ts'].min(), "->", train['created_ts'].max())

print()
print("=== TEST SPLIT ===")
print("  Rows:", len(test))
print("  Positive (label=1):", int(test['label'].sum()))
print("  Negative (label=0):", int((test['label']==0).sum()))
rate_te = test['label'].mean()*100
print("  Positive rate: {:.2f}%".format(rate_te))
print("  Date range:", test['created_ts'].min(), "->", test['created_ts'].max())

print()
print("=== OVERALL ===")
print("  Total positive:", int(df['label'].sum()))
print("  Total negative:", int((df['label']==0).sum()))
print("  Overall date range:", df['created_ts'].min(), "->", df['created_ts'].max())
