#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:04:26 2023

@author: isaac
"""

import sys
from datetime import datetime

def calculate_days_difference(date_str1, date_str2):
    # Convert date strings to datetime objects
    date1 = datetime.strptime(date_str1, '%Y%m%d')
    date2 = datetime.strptime(date_str2, '%Y%m%d')

    # Calculate the difference in days
    days_difference = abs((date2 - date1).days)

    return days_difference

if __name__ == "__main__":
    # Check if two date strings are provided as command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py yyyymmdd1 yyyymmdd2")
        sys.exit(1)

    date_str1 = sys.argv[1]
    date_str2 = sys.argv[2]

    try:
        # Calculate and print the number of days between the two dates
        days_difference = calculate_days_difference(date_str1, date_str2)
        print(f"Number of days between {date_str1} and {date_str2}: {days_difference} days")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
