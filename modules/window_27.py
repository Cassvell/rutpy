from datetime import datetime, timedelta
from modules.calc_daysdiff import calculate_days_difference
import math

def window_27(idate, fdate, format):
    ndays = calculate_days_difference(idate, fdate)

    idate = datetime.strptime(idate + ' 00:00:00', '%Y%m%d %H:%M:%S')
    fdate = datetime.strptime(fdate + ' 23:59:00', '%Y%m%d %H:%M:%S')

    nwindows = math.ceil(ndays/27)
    
    iwindows = []
    fwindows = []
    medwindows=[]
    for w in range(nwindows):
        window_start = idate + timedelta(days=w * 27)
        window_end = window_start + timedelta(days=26, hours=23, minutes=59)
        window_med = window_start + timedelta(days=12, hours=23, minutes=59)
        
        if window_end >= fdate:
            window_end = fdate
        #print(f'\n Window {w+1}: {(window_start)} to {(window_end)} \n')
        doi_start = window_start.timetuple().tm_yday
        doi_med = window_med.timetuple().tm_yday
        doi_end = window_end.timetuple().tm_yday
        year_start = window_start.year
        year_med = window_med.year
        year_end = window_end.year
        
        #print(f'Window {w+1}: {year_start:04d}-{doi_start:03d} to {year_end:04d}-{doi_end:03d}')
        tmp_idate = f"{year_start:04d}-{doi_start}"
        tmp_fdate = f"{year_end:04d}-{doi_end}"
        tmp_meddate = f"{year_med:04d}-{doi_med}"
        if format == 'doy':
            iwindows.append(tmp_idate)
            fwindows.append(tmp_fdate)
            medwindows.append(tmp_meddate)
            
        elif format == 'date':
            iwindows.append(window_start)
            fwindows.append(window_end)
            medwindows.append(window_med)            
    return iwindows, medwindows, fwindows, nwindows