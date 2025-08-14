from core.utils import load_csv_parse_datetime
p = "data/MES-0612-0806.Last_3min.csv"
df = load_csv_parse_datetime(p, tz_source=None, tz_target='America/New_York')
print("Cols:", df.columns.tolist())
print("Rows:", len(df))
print("Index tz:", df.index.tz)
print("First rows:\n", df.head().to_string())
print("08:00-12:00 session bars:", df.between_time("08:00","12:00").shape[0])