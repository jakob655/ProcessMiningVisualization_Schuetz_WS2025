import pandas as pd
import csv
import os

from api.custom_error import BadColumnException

'''
For reading in csv files. Returns a list of all cases.
'''


def read(filename, timeLabel='timestamp', caseLabel='case', eventLabel='event'):
    possible_delimiters = [',', ';', '\t', '|', ' ', ':']
    df = None
    delimiter = None

    # Try to detect the delimiter using csv.Sniffer
    with open(filename, 'r', encoding='utf-8-sig') as f:
        sample = f.read(1024)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except Exception:
            # If csv.Sniffer fails, try each possible delimiter
            for possible_delimiter in possible_delimiters:
                try:
                    df = pd.read_csv(filename, delimiter=possible_delimiter, encoding='utf-8-sig')
                    if all(col in df.columns for col in [timeLabel, caseLabel, eventLabel]):
                        delimiter = possible_delimiter
                        break
                except pd.errors.ParserError:
                    continue

    if delimiter is None:
        raise ValueError("Could not determine delimiter")

    # If delimiter is detected using csv.Sniffer, read the CSV file
    if df is None:
        df = pd.read_csv(filename, delimiter=delimiter, encoding='utf-8-sig')

    # Check that the required columns exist
    required_columns = [timeLabel, caseLabel, eventLabel]
    if not all(col in df.columns for col in required_columns):
        raise BadColumnException("csv_preprocessor.py: ERROR: Selected columns not found in DataFrame")

    # Sort by timestamp
    df = df.sort_values(by=[caseLabel, timeLabel])

    # Create a dictionary to store the events for each case
    cases = {}

    for _, row in df.iterrows():
        case = row[caseLabel]
        event = row[eventLabel]

        if case in cases:
            cases[case].append(event)
        else:
            cases[case] = [event]

    array = list(cases.values())

    # Return the list of cases
    return array


# DEPRECATED: now using pickle instead
def save(filename, cases):
    # Save the cases, so it can be loaded in future sessions without read_cases() again:
    array = cases

    name = os.path.splitext(os.path.basename(filename))[0]
    destination_path = "saves/"
    destination = destination_path + name + ".txt"

    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination))

    with open(destination, "w") as f:
        for i, case in enumerate(array):
            for j, event in enumerate(case):
                if j > 0:
                    f.write(",")
                f.write(str(event))
            if i < len(array) - 1:
                f.write("\n")


# DEPRECATED: now using pickle instead
def read_cases(filename):
    log = []
    #cwd = os.getcwd()
    #path = os.path.join(cwd, filename)
    with open(filename, 'r') as f:
        for line in f.readlines():
            assert isinstance(line, str)
            log.append(list(line.split(",")))
    return log
