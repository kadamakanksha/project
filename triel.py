import xlsxwriter
 
# Workbook() takes one, non-optional, argument
# which is the filename that we want to create.
workbook = xlsxwriter.Workbook('hello1.xlsx')
 
# The workbook object is then used to add new
# worksheet via the add_worksheet() method.
worksheet = workbook.add_worksheet()
 
# Use the worksheet object to write
# data via the write() method.
worksheet.write('A1', '1')
worksheet.write('B1', 'Saurabh')
worksheet.write('C1', 'Yes')


worksheet.write('A2', '2')
worksheet.write('B2', 'Swastik')
worksheet.write('C2', 'Yes')

worksheet.write('A3', '3')
worksheet.write('B3', 'Samruddhi')
worksheet.write('C3', 'Yes')
 
# Finally, close the Excel file
# via the close() method.
workbook.close()