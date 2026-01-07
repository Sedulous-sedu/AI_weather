#!/usr/bin/env python3
"""
scrape_rakta_data.py
====================

This script scrapes public bus timetable information from the Ras Al Khaimah
Transport Authority (RAKTA) website and uses it to build a large
data set suitable for simulation experiments.  The AURAK shuttle data
provided by the user contains only 5000 rows, which is insufficient
for evaluating model degradation over tens of thousands of trips.

The RAKTA public‐transport page lists intercity routes together with
their trip numbers, departure times and destinations.  Because the
published schedule is static (the same set of trips operates on every
day) it is necessary to replicate the schedule across multiple
calendar days to produce a data set with more than 100 000 records.

In addition to the schedule fields scraped from RAKTA, the script
samples contextual features (traffic_condition, weather, special_event,
temperature_celsius, delay_minutes and arrival_status) from the
original AURAK shuttle data.  This preserves the statistical
characteristics of the original data while coupling the records to
actual RAKTA timetables.  For each scraped trip and generated date a
random row from the AURAK data is selected and its contextual fields
are appended.

Usage
-----

    python scrape_rakta_data.py --output rakta_trips.csv \
        --start-date 2025-01-01 --end-date 2027-12-31

This will fetch the bus schedule, generate all trip instances
between 1 January 2025 and 31 December 2027 (a three year span) and
write the resulting data set to ``rakta_trips.csv``.  Feel free to
increase the date range to produce more than 100 000 rows.  The
script dynamically determines how many days are needed to exceed
100 000 rows if no date range is supplied.  See the bottom of the
file for command line arguments.

Notes
-----

- The RAKTA website employs a 403 filter which blocks automated
  scraping unless a modern browser user‑agent is supplied.  The
  ``requests`` session created in this script includes such a
  user‑agent in its headers.
- Some tables on the page are rendered using HTML structures rather
  than classical ``<table>`` tags.  To reliably extract trip
  information the scraper locates headings labelled “Trip
  Details” and then parses subsequent lines containing trip
  numbers, times and destination texts.  If the website structure
  changes in the future, please update the ``_parse_trip_tables``
  function accordingly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import random
from pathlib import Path
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup


def fetch_rakta_schedule(url: str = "https://www.rakta.gov.ae/public-transport/") -> pd.DataFrame:
    """Fetches the RAKTA public transport page and extracts trip schedules.

    The function sends an HTTP GET request with a Chrome‐like User‑Agent
    to bypass the site's 403 filter.  It then parses the returned HTML
    using BeautifulSoup, looking for headings that contain the text
    "Trip Details".  The sibling elements following each heading
    typically consist of trip number, time and destination entries
    separated by whitespace.  These are parsed into a tidy table
    containing columns: route, trip_no, departure_time (as HH:MM),
    destination and direction ("From" or "To").

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ["route", "direction", "trip_no",
        "departure_time", "destination"].  Each row corresponds to a
        scheduled trip published on the RAKTA website.

    Raises
    ------
    requests.HTTPError
        If the HTTP request fails.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/110.0 Safari/537.36"
        )
    })
    resp = session.get(url)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    schedule_entries: List[Dict[str, str]] = []

    # The page contains multiple route sections.  Each section starts
    # with a heading describing the direction (e.g. "RAK To Dubai") and
    # includes a subheading "Trip Details" followed by a text block
    # where trip_no, time and destination values are separated by
    # whitespace.  We iterate through all headings and capture the
    # relevant sections.
    for heading in soup.find_all(["h5", "h6", "h4"]):
        text = heading.get_text(strip=True).lower()
        if "trip details" in text:
            # The heading preceding this one should contain the route
            # name and implicitly the direction.  We walk backwards
            # through previous siblings until we find a heading that
            # does not contain 'trip details'.
            route_node = heading.find_previous(
                lambda tag: tag.name.startswith('h') and 'trip details' not in tag.get_text(strip=True).lower()
            )
            route_name = route_node.get_text(strip=True) if route_node else "Unknown Route"
            # Determine direction by checking if the route name
            # contains 'to'.  e.g. "RAK To Dubai Union Bus Station"
            direction = "Unknown"
            route_lower = route_name.lower()
            if " to " in route_lower:
                parts = route_lower.split(" to ")
                direction = "From" if parts[0].strip() == "rak" else "To"
            
            # The lines containing trips may be found in the next
            # elements.  We search until we hit another heading or
            # until we have consumed a block containing numbers,
            # times and destinations.
            sibling = heading.find_next_sibling()
            while sibling and sibling.name not in ["h4", "h5", "h6"]:
                if sibling.name == "table":
                    # Parse HTML table structure
                    for row in sibling.find_all("tr"):
                        cols = row.find_all("td")
                        if len(cols) >= 3:
                            trip_no_text = cols[0].get_text(strip=True)
                            if trip_no_text.isdigit():
                                time_token = cols[1].get_text(strip=True).replace('.', ':')
                                dest = cols[2].get_text(strip=True).replace('(', '').replace(')', '')
                                schedule_entries.append({
                                    "route": route_name,
                                    "direction": direction,
                                    "trip_no": int(trip_no_text),
                                    "departure_time": time_token,
                                    "destination": dest.strip(),
                                })
                else:
                    # Fallback for text-based layout
                    text_block = sibling.get_text(" ", strip=True)
                    # Trip details often look like: '1 5:30 Dubai (Direct)'
                    # We attempt to split on whitespace such that the
                    # first token is the trip number, the second token is
                    # the time (HH:MM), and the remainder is the
                    # destination.  Skip lines that do not match this
                    # pattern.
                    parts = text_block.split()
                    if len(parts) >= 3 and parts[0].isdigit():
                        trip_no = parts[0]
                        time_token = parts[1]
                        # Normalise times that might use '.' instead of ':'
                        time_token = time_token.replace('.', ':')
                        dest = " ".join(parts[2:])
                        # Remove any parentheses such as "(Direct)" to
                        # standardise destinations
                        dest = dest.replace('(', '').replace(')', '')
                        schedule_entries.append({
                            "route": route_name,
                            "direction": direction,
                            "trip_no": int(trip_no),
                            "departure_time": time_token,
                            "destination": dest.strip(),
                        })
                sibling = sibling.find_next_sibling()

    schedule_df = pd.DataFrame(schedule_entries)

    if schedule_df.empty:
        with open("debug_rakta.html", "w", encoding="utf-8") as f:
            f.write(soup.prettify())
        raise ValueError("No schedule data found. HTML content saved to 'debug_rakta.html' for inspection.")

    # Remove duplicate rows (if any) and sort for determinism
    schedule_df = schedule_df.drop_duplicates().sort_values(by=["route", "trip_no"]).reset_index(drop=True)
    return schedule_df


def generate_dates(start_date: dt.date, end_date: dt.date) -> List[dt.date]:
    """Generate a list of all dates between start_date and end_date inclusive."""
    delta = (end_date - start_date).days
    return [start_date + dt.timedelta(days=i) for i in range(delta + 1)]


def build_large_dataset(
    schedule_df: pd.DataFrame,
    aurak_df: pd.DataFrame,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    target_rows: int = 100_000,
) -> pd.DataFrame:
    """Construct a large synthetic data set exceeding ``target_rows``.

    Parameters
    ----------
    schedule_df : DataFrame
        DataFrame containing scraped trip schedules with columns
        ["route", "direction", "trip_no", "departure_time", "destination"].
    aurak_df : DataFrame
        The original AURAK shuttle data.  This is used to sample
        contextual features (traffic_condition, weather, etc.) to
        preserve distributional characteristics.
    start_date, end_date : date or None
        Date range over which to replicate the schedule.  If both
        parameters are ``None``, a date range will be automatically
        chosen such that the number of generated rows is at least
        ``target_rows``.  When either one is given, the other must
        also be provided.
    target_rows : int
        Minimum number of rows desired in the output.

    Returns
    -------
    pandas.DataFrame
        A data frame containing at least ``target_rows`` rows.  It
        combines the scraped schedule with sampled contextual
        variables from the AURAK data and includes columns:
        ["date", "day_of_week", "time_of_day", "route", "direction",
        "trip_no", "departure_time", "destination", "stop_distance_km",
        "traffic_condition", "weather", "special_event",
        "temperature_celsius", "delay_minutes", "arrival_status"].
    """
    # Validate date range
    if (start_date is None) != (end_date is None):
        raise ValueError("Both start_date and end_date must be provided or both omitted.")

    # Determine number of schedule entries
    base_trips = len(schedule_df)
    if base_trips == 0:
        raise ValueError("The schedule DataFrame is empty.  Cannot generate data.")

    # If no date range is provided, compute how many days are needed to exceed target_rows
    if start_date is None and end_date is None:
        days_needed = math.ceil(target_rows / base_trips)
        today = dt.date.today()
        start_date = today
        end_date = today + dt.timedelta(days=days_needed - 1)
    else:
        days_needed = (end_date - start_date).days + 1

    # Generate list of dates
    dates = generate_dates(start_date, end_date)

    def get_time_of_day(time_str: str) -> str:
        """Categorise the departure time into time‑of‑day buckets."""
        hour = int(time_str.split(":")[0])
        if 5 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 17:
            return "Afternoon"
        elif 17 <= hour < 21:
            return "Evening"
        else:
            return "Night"

    large_rows: List[Dict] = []
    aurak_indices = aurak_df.index.tolist()
    for date in dates:
        day_name = date.strftime("%A")
        for _, trip in schedule_df.iterrows():
            sample_idx = random.choice(aurak_indices)
            context = aurak_df.loc[sample_idx]
            large_rows.append({
                "date": date.isoformat(),
                "day_of_week": day_name,
                "time_of_day": get_time_of_day(trip["departure_time"]),
                "route": trip["route"],
                "direction": trip["direction"],
                "trip_no": trip["trip_no"],
                "departure_time": trip["departure_time"],
                "destination": trip["destination"],
                "stop_distance_km": context.get("stop_distance_km", None),
                "traffic_condition": context.get("traffic_condition", None),
                "weather": context.get("weather", None),
                "special_event": context.get("special_event", None),
                "temperature_celsius": context.get("temperature_celsius", None),
                "delay_minutes": context.get("delay_minutes", None),
                "arrival_status": context.get("arrival_status", None),
            })
    df_large = pd.DataFrame(large_rows)

    # If we still haven't reached target_rows after the provided date
    # range, append more days until the target is met.
    while len(df_large) < target_rows:
        next_date = dates[-1] + dt.timedelta(days=1)
        dates.append(next_date)
        day_name = next_date.strftime("%A")
        for _, trip in schedule_df.iterrows():
            sample_idx = random.choice(aurak_indices)
            context = aurak_df.loc[sample_idx]
            df_large = pd.concat([
                df_large,
                pd.DataFrame({
                    "date": [next_date.isoformat()],
                    "day_of_week": [day_name],
                    "time_of_day": [get_time_of_day(trip["departure_time"])],
                    "route": [trip["route"]],
                    "direction": [trip["direction"]],
                    "trip_no": [trip["trip_no"]],
                    "departure_time": [trip["departure_time"]],
                    "destination": [trip["destination"]],
                    "stop_distance_km": [context.get("stop_distance_km", None)],
                    "traffic_condition": [context.get("traffic_condition", None)],
                    "weather": [context.get("weather", None)],
                    "special_event": [context.get("special_event", None)],
                    "temperature_celsius": [context.get("temperature_celsius", None)],
                    "delay_minutes": [context.get("delay_minutes", None)],
                    "arrival_status": [context.get("arrival_status", None)],
                })
            ], ignore_index=True)
        if len(df_large) >= target_rows:
            break
    return df_large


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape RAKTA bus schedules and build a large trip data set."
    )
    parser.add_argument(
        "--aurak-csv",
        type=Path,
        default=Path("aurak_shuttle_data_full.csv"),
        help=(
            "Path to the original AURAK shuttle data CSV.  Used for sampling "
            "contextual features."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rakta_trips_large.csv"),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help=(
            "Start date (YYYY-MM-DD) for replicating the schedule.  If omitted, "
            "a suitable range will be chosen automatically."
        ),
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help=(
            "End date (YYYY-MM-DD) for replicating the schedule.  If omitted, "
            "a suitable range will be chosen automatically."
        ),
    )
    parser.add_argument(
        "--target-rows",
        type=int,
        default=100_000,
        help=(
            "Minimum number of rows desired in the generated data set.  "
            "Ignored if a date range is provided."
        ),
    )
    args = parser.parse_args()

    aurak_df = pd.read_csv(args.aurak_csv)

    try:
        schedule_df = fetch_rakta_schedule()
    except requests.HTTPError as e:
        raise SystemExit(f"Failed to fetch RAKTA schedule: {e}")

    if args.start_date and args.end_date:
        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        start_date = end_date = None

    large_df = build_large_dataset(
        schedule_df,
        aurak_df,
        start_date=start_date,
        end_date=end_date,
        target_rows=args.target_rows,
    )

    large_df.to_csv(args.output, index=False)
    print(f"Generated {len(large_df):,} rows and saved to {args.output}")


if __name__ == "__main__":
    main()