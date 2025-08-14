# check_breakouts.py
import pandas as pd

p = "data/MES-0612-0806.Last_3min.csv"
print("Reading", p)
df = pd.read_csv(p, parse_dates=['DateTime'])
df.columns = [c.strip() for c in df.columns]
df = df.set_index('DateTime').sort_index()

# If timestamps are naive but represent ET, localize to ET
if df.index.tz is None:
    try:
        df.index = df.index.tz_localize('America/New_York')
    except Exception:
        # if they are UTC, you may need tz_localize('UTC').tz_convert('America/New_York')
        pass

print("Overall rows:", len(df), "index min/max:", df.index.min(), df.index.max())

# Check per-day opening-range breakouts (08:00-12:00 session)
for day, day_df in df.groupby(df.index.date):
    # select session bars 08:00-12:00 local time
    session = day_df.between_time("08:00", "12:00")
    if session.empty:
        print(day, " -> no session bars")
        continue
    bars_per_or = int(15 / 3)  # 3-min bars
    if len(session) <= bars_per_or:
        print(day, " -> not enough bars for OR (len=%d)" % len(session))
        continue
    or_bars = session.iloc[:bars_per_or]
    or_high = or_bars['High'].max()
    or_low = or_bars['Low'].min()
    post = session.iloc[bars_per_or:]
    breakouts = post[(post['Close'] > or_high) | (post['Close'] < or_low)]
    print(day, "OR_high/low:", or_high, or_low, "post_bars:", len(post), "raw_breakouts:", len(breakouts))
    if not breakouts.empty:
        print(breakouts.head().to_string())