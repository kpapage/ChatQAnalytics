from datetime import date,timedelta
searchdate = "2023-03-17..2023-03-20"
today = date.today()
d = timedelta(days=7)
week = today + d
newsearchdate = str(today) + '...' + str(week)
print(newsearchdate)
