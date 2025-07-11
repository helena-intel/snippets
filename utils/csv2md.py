#!/usr/bin/env python3

# Display csv data in markdown table format
# Usage: `python csv2md.py /path/to/data.csv`
# Requirements: `pip install pandas tabulate`

import pandas as pd
import argparse

def csv_to_markdown_table(file_path, columns=None):
    df = pd.read_csv(file_path)

    if columns:
        df = df[columns]

    markdown_table = df.to_markdown(index=False)

    return markdown_table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert a CSV file to a Markdown table.')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('-c', '--columns', nargs='+', help='A list of column names to include in the Markdown table')
    args = parser.parse_args()

    markdown_output = csv_to_markdown_table(args.csv_file, columns=args.columns)
    print(markdown_output)
