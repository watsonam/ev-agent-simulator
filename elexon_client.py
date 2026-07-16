from datetime import date, timedelta
from pathlib import Path
from uuid import uuid4

import requests
import pandas as pd

class ElexonClient:
    BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

    def get_market_index(self, from_date: str, to_date: str | None = None) -> pd.DataFrame | None:
        if to_date is None:
            to_date = from_date

        from_timestamp = f"{from_date}T00:00Z"
        to_timestamp = f"{to_date}T00:00Z"

        response = requests.get(
            f"{self.BASE_URL}/balancing/pricing/market-index",
            params={
                "from": from_timestamp,
                "to": to_timestamp,
                "dataProviders": "APXMIDP",
                "settlementPeriodFrom": 1,
                "settlementPeriodTo": 48
            },
            timeout=30
        )

        if response.status_code == 200 and response.text.strip():
            data = response.json()
            if isinstance(data, dict) and "data" in data:
                df = pd.DataFrame(data["data"])
            else:
                df = pd.DataFrame(data)
            if "settlementPeriod" in df.columns:
                df["hour"] = ((df["settlementPeriod"] - 1) // 2).astype(int)
            return df
        return None


BOOTSTRAP_DAYS = 180


def update_day_ahead_prices(csv_path: Path) -> date | None:
    # The committed CSV is the fallback: if the file is missing I bootstrap it,
    # and if Elexon is unreachable I just keep what's on disk rather than crash.
    # I only write when there are genuinely new rows, and write to a temp file
    # then rename, so a reader never catches the CSV half-written or missing.
    try:
        existing = pd.read_csv(csv_path)
        last_date = pd.to_datetime(existing["settlementDate"]).max().date()
    except (FileNotFoundError, ValueError, KeyError, pd.errors.EmptyDataError):
        existing = pd.DataFrame()
        last_date = date.today() - timedelta(days=BOOTSTRAP_DAYS)

    today = date.today()
    target = today + timedelta(days=1)
    start = today if not existing.empty else last_date

    client = ElexonClient()
    new_frames = []
    d = start
    while d <= target:
        try:
            df = client.get_market_index(d.strftime("%Y-%m-%d"))
        except Exception:
            df = None
        if df is not None and not df.empty:
            new_frames.append(df)
        d += timedelta(days=1)

    if not new_frames:
        return last_date if not existing.empty else None

    combined = pd.concat([existing, *new_frames], ignore_index=True)
    combined = combined.drop_duplicates(subset=["settlementDate", "settlementPeriod"], keep="last")
    combined = combined.sort_values(["settlementDate", "settlementPeriod"])

    if len(combined) == len(existing):
        return pd.to_datetime(combined["settlementDate"]).max().date()

    tmp = csv_path.with_suffix(f".{uuid4().hex}.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(csv_path)
    return pd.to_datetime(combined["settlementDate"]).max().date()


if __name__ == "__main__":
    from datetime import datetime

    client = ElexonClient()
    end_date = datetime(2026, 7, 7)
    start_date = end_date - timedelta(days=180)

    print(f"Fetching {start_date.date()} to {end_date.date()}")

    all_data = []
    current_date = start_date
    count = 0

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        print(date_str)
        df = client.get_market_index(date_str)
        if df is not None:
            all_data.append(df)
            count += len(df)

        current_date += timedelta(days=1)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Total: {len(combined_df)} records")
        combined_df.to_csv("market_index.csv", index=False)
        print("Exported to market_index.csv")
    else:
        print("No data")
    