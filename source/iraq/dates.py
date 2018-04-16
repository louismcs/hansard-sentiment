from datetime import date
from datetime import timedelta


DB_PATH = 'Data/corpus.db'
START_DATE = date(2001, 9, 11)
END_DATE = date(2003, 3, 19)


def date_range(start_date, end_date):
    """Returns all dates between start_date (inclusive) and end_date (exclusive)"""
    for count in range(int((end_date - start_date).days)):
        yield start_date + timedelta(count)
