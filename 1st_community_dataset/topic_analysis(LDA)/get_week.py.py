from time import strptime
import pandas as pd
from datetime import datetime

def get_week(date):
    date_format = datetime.strptime(date, "%Y-%m-%d")
    week = date_format.isocalendar()[1]
    return week


print(get_week('2020-12-14'))