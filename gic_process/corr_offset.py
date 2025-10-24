#import os
import sys
import numpy as np
#from datetime import datetime, timedelta
#import pandas as pd
from gicdproc import  process_station_data
from calc_daysdiff import calculate_days_difference
import matplotlib.pyplot as plt
from gic_threshold import threshold



def corr_offset(data, threshold, window_size, stddev):
 # Size of the moving window in minutes
    ndata = len(data)
    crossing_indices = []
    median_values = []
    resampled_data = int(ndata/window_size)
    
    threshold_level = threshold - stddev  # Adjust threshold if needed
    for i in range(resampled_data):
        window = data[i*window_size:(i + 1)*window_size]
        
        window_median = np.nanmedian(np.abs(window))
        
        median_values.append(window_median)
        
    for i in range(1, len(median_values)):
        prev_median = median_values[i-1]
        current_median = median_values[i]
        
        # Skip if either median is NaN
        if np.isnan(prev_median) or np.isnan(current_median):
            continue
        
        # Detect crossings with hysteresis (more robust)
        crossing_up = (prev_median <= threshold_level) and (current_median > threshold_level)
        crossing_down = (prev_median >= threshold_level) and (current_median < threshold_level)
        
        if crossing_up or crossing_down:
            # Convert resampled index back to original data index
            original_index = i * window_size
            crossing_indices.append(original_index)

        
    data_corr_offset = data.copy()
    if crossing_indices:   
        c = [0,0]
            
        if np.abs(np.nanmedian(data[0:59])) <= threshold_level and np.abs(np.nanmedian(data[-59:])) <= threshold_level:
            c = [0,0]
        elif np.abs(np.nanmedian(data[0:59])) >= threshold_level and np.abs(np.nanmedian(data[-59:])) >= threshold_level:
            c = [1,1]
        
        if np.abs(np.nanmedian(data[0:59])) >= threshold_level and np.abs(np.nanmedian(data[-59:])) <= threshold_level:
            c = [1,0]
            
        elif np.abs(np.nanmedian(data[0:59])) <= threshold_level and np.abs(np.nanmedian(data[-59:])) >= threshold_level:
            c = [0,1]

            
        print(f'Number of threshold crossings: {(crossing_indices)}, Case: {c}')
        #caso 0, crossing

        if c ==[0,0]:
            if len(crossing_indices) == 2:
                start_idx = crossing_indices[0]
                end_idx = crossing_indices[1]
                
                sampled_data = data[start_idx:end_idx]
                median_w = np.nanmedian(sampled_data)
                data_corr_offset[start_idx:end_idx] = sampled_data - median_w
                
            else: 
                
                for i in range(len(crossing_indices)):
                    idx = crossing_indices[i]
                    start_idx = idx
                    end_idx = idx+1
                    sampled_data = data[start_idx:end_idx]
                    median_w = np.nanmedian(sampled_data)
                    data_corr_offset[start_idx:end_idx] = sampled_data - median_w
            
    
                    

        #caso si, el numero de indices es impar y el final de la ventana muestra offset alterado
        elif c ==[0,1]:
            
            if len(crossing_indices) == 1:
                sampled_data = data[crossing_indices[0]:]
                median_w = np.nanmedian(data[crossing_indices[0]:])
                data_corr_offset[crossing_indices[0]:] = sampled_data - median_w
            elif len(crossing_indices) > 1:
                for i in range(len(crossing_indices)):
                    idx = crossing_indices[i]
                    if i < len(crossing_indices) - 1:
                        start_idx = idx
                        end_idx = idx+1
                        sampled_data = data[start_idx:end_idx]
                        median_w = np.nanmedian(data[start_idx:end_idx])    
                        data_corr_offset[start_idx:end_idx] = sampled_data - median_w
                        
                    else:
                        sampled_data = data[idx:]
                        median_w = np.nanmedian(data[idx:])    
                        data_corr_offset[idx:] = sampled_data - median_w
            
        elif c == [1,0]:
            if len(crossing_indices) == 1:
                sampled_data = data[:crossing_indices[0]]
                median_w = np.nanmedian(data[:crossing_indices[0]])
                data_corr_offset[:crossing_indices[0]] = sampled_data - median_w

            
            elif len(crossing_indices) > 1:
                for i in range(len(crossing_indices)):
                    idx = crossing_indices[i]
                    if i == 0:
                        sampled_data = data[:idx]
                        median_w = np.nanmedian(data[:idx])
                        data_corr_offset[:idx] = sampled_data - median_w
                    else:
                        start_idx = idx
                        end_idx = idx+1
                        
                        sampled_data = data[start_idx:end_idx]
                        median_w = np.nanmedian(data[start_idx:end_idx])
                        data_corr_offset[start_idx:end_idx] = sampled_data - median_w                
        #caso cuando el inicio y el final de la ventana presentan un offset alterado
        elif c == [1,1]:        
            if len(crossing_indices) == 2:
                start_idx = crossing_indices[0]
                end_idx = crossing_indices[1]
                
                # Process all three segments
                segments = [
                    (0, start_idx),           # Beginning
                    (start_idx, end_idx),      # Middle  
                    (end_idx, len(data))       # End
                ]
                
                for seg_start, seg_end in segments:
                    if seg_start < seg_end:  # Only process valid segments
                        segment_data = data[seg_start:seg_end]
                        if len(segment_data) > 0:  # Only if segment has data
                            median_w = np.nanmedian(segment_data)
                            data_corr_offset[seg_start:seg_end] = segment_data - median_w
                
            else:
                for i in range(len(crossing_indices) + 1):  # +1 to include segment after last crossing
                    if i == 0:
                        # First segment: from start to first crossing
                        seg_start = 0
                        seg_end = crossing_indices[0]
                    elif i == len(crossing_indices):
                        # Last segment: from last crossing to end
                        seg_start = crossing_indices[-1]
                        seg_end = len(data)
                    else:
                        # Middle segments: between crossings
                        seg_start = crossing_indices[i-1]
                        seg_end = crossing_indices[i]
                    
                    # Process the segment
                    if seg_start < seg_end:
                        segment_data = data[seg_start:seg_end]
                        if len(segment_data) > 0:
                            median_w = np.nanmedian(segment_data)
                            data_corr_offset[seg_start:seg_end] = segment_data - median_w                     
                # Initialize output array or list
                
            
                        
    else:
        print('No threshold crossings detected. No offset correction applied.')
    
    return(data_corr_offset)