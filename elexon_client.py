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


def update_day_ahead_prices(csv_path: Path) -> None:
    existing = pd.read_csv(csv_path)
    last_date = pd.to_datetime(existing["settlementDate"]).max().date()
    today = date.today()

    client = ElexonClient()
    new_frames = []
    d = min(today, last_date)
    while d <= today + timedelta(days=1):
        df = client.get_market_index(d.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            new_frames.append(df)
        d += timedelta(days=1)

    if not new_frames:
        return

    combined = pd.concat([existing, *new_frames], ignore_index=True)
    combined = combined.drop_duplicates(subset=["settlementDate", "settlementPeriod"], keep="last")
    combined = combined.sort_values(["settlementDate", "settlementPeriod"])
    if len(combined) == len(existing):
        return

    tmp = csv_path.with_suffix(f".{uuid4().hex}.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(csv_path)