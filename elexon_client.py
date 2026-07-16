from datetime import date, timedelta
from pathlib import Path

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


def update_day_ahead_prices(csv_path: Path) -> date:
    """Fetch any missing days between the CSV's latest date and tomorrow
    (day-ahead prices for tomorrow are normally published this afternoon)
    and append them. Cheap to call on every dashboard run: short-circuits
    with no network call once the CSV is already up to date.

    Fetching through tomorrow, not just today, matters for Intelligent
    Octopus: its charging schedule looks up to a day past "now" to plan an
    overnight charge, so without tomorrow's prices already in the CSV,
    simulating "today" would fail partway through the evening."""
    existing = pd.read_csv(csv_path)
    last_date = pd.to_datetime(existing["settlementDate"]).max().date()
    target = date.today() + timedelta(days=1)
    if last_date >= target:
        return last_date

    client = ElexonClient()
    new_frames = []
    d = last_date + timedelta(days=1)
    while d <= target:
        df = client.get_market_index(d.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            new_frames.append(df)
        d += timedelta(days=1)

    if not new_frames:
        return last_date

    combined = pd.concat([existing, *new_frames], ignore_index=True)
    combined = combined.drop_duplicates(subset=["settlementDate", "settlementPeriod"], keep="last")
    combined = combined.sort_values(["settlementDate", "settlementPeriod"])
    combined.to_csv(csv_path, index=False)
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
    