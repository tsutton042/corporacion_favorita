"""
Script to do the initial merge of the data.
Designed to only be run once, so that this processing doesn't need to be done again.
Was planning to use sqlite3 so we only need to load the relevant data at any given point,
    but this was quicker to write (even if a bit more janky)
"""

import pandas as pd
from datetime import date
import os
from tqdm.auto import tqdm


def __merge_dataframes(df_dict, kind):
    """
    Helper function to create train/test dataframes
    """
    assert kind in ["test", "train"], "Invalid kind argument!"
    df = df_dict[kind]
    df = df.join(df_dict["items"], on="item_nbr", how="left")
    df = df.join(df_dict["stores"], on="store_nbr", how="left")
    df = df.join(df_dict["transactions"], on=["store_nbr", "date"], how="left")
    df = df.join(
        df_dict["oil"], on="date", how="left"
    )  # this will need later processing
    # Process holiday data
    hols = df_dict["holidays_events"]
    # > Matching holiday types
    # * if the day is a weekday, mark as a working day, if day is weekend, mark as day off
    # Then consider holidays as below:
    # * if type is "Holiday" and transferred is True, find the date it's transferred to
    #   * these are marked as "Transfer" with description being "Traslado {holiday_name}"
    #   * Approach: Ignore this date, move to next date - handle with "Transfer" case
    # * if type is in ["Bridge", "Additional", "Transfer", "Event"]  or
    #     (type is "Holiday" and transferred is False), assign the day as a day off
    # * if type is "Work Day", assign the day as a working day
    # > Merging on locations
    # * if locale == "National" (locale_name == "Ecuador"), match to every store
    # * if locale == "Regional", match locale_name to store state
    # * if locale == "Local", match locale_name to store city
    work_day = [
        x.weekday() <= 4 and x.weekday() >= 0 for x in df.date.map(date.fromisoformat)
    ]
    df["work_day"] = work_day
    df["day_off"] = ~df.work_day
    for idx in hols.index:
        day = hols.loc[idx]
        # get affected df indexes
        if day.locale == "National":
            cond = df.date == day.date
        elif day.locale == "Regional":
            cond = (df.date == day.date) & (df.state == day.locale_name)
        elif day.locale == "Local":
            cond = (df.date == day.date) & (df.city == day.locale_name)
        df_idxs = df[cond].idxs
        # set day_off & work day based on day.type
        # I want to code golf this, it feels messy and hard to read, but so does one-liner approach
        if day.type == "Work Day":
            df.loc[df_idxs, "work_day"] = True
            df.loc[df_idxs, "day_off"] = False
        elif day.type in ["Bridge", "Additional", "Transfer", "Event"] or (
            day.type == "Holiday" and not day.transferred
        ):
            df.loc[df_idxs, "work_day"] = False
            df.loc[df_idxs, "day_off"] = True
        else:
            # (day.type == "Holiday" and day.transferred), do nothing
            pass
    return df


def get_data(data_dir=f"{os.path.dirname(__file__)}/data") -> None:
    """
    Process the CSVs and put the resulting data into a single pandas DataFrame.
    Only intended to be run once, as the dataframe will be saved
    """
    print("Reading CSVs")
    csvs = [fn for fn in os.listdir(data_dir) if fn[-4:] == ".csv"]
    data_frames = {}
    for file in csvs:
        if file != "sample_submission.csv":
            table_name = file[:-4]
            print(f" *  Reading {file}")
            # memory bottlenecked, probably due to train containing 125m rows
            # could write a custom file reader function that's more memory efficient,
            # but that seems like more work than it's worth
            with tqdm() as bar:
                data_frames[table_name] = pd.read_csv(
                    f"{data_dir}/{file}",
                    engine="python",  # c was crashing
                    skiprows=lambda _: bar.update(1)
                    and False,  # messy but useful progress bar
                )
    # get train & test dataframes
    print("Merging train data")
    train = __merge_dataframes(data_frames, "train")
    print("Merging test data")
    test = __merge_dataframes(data_frames, "test")
    print("Saving to disk!")
    train.to_csv(f"{data_dir}/test_merged.csv", index=False)
    test.to_csv(f"{data_dir}/train_merged.csv", index=False)


if __name__ == "__main__":
    get_data()
