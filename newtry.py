# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 02:43:09 2023

@author: COMPUTER
"""
from openpyxl import Workbook
workbook = Workbook()
spreadsheet = workbook.active

spreadsheet["A1"] = "1"
spreadsheet["B1"] = "Saurabh"
spreadsheet["C1"] = "present"


spreadsheet["A2"] = "2"
spreadsheet["B2"] = "Swastik"
spreadsheet["C2"] = "present"

spreadsheet["A3"] = "3"
spreadsheet["B3"] = "Siddhi"
spreadsheet["C3"] = "Absent"

spreadsheet["A4"] = "4"
spreadsheet["B4"] = "Akanksha"
spreadsheet["C4"] = "present"

workbook.save(filename="hello1.xlsx")