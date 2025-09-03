from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def on_move(event):
    
        print(f'data coords {event.xdata} {event.ydata},',
              f'pixel coords {event.x} {event.y}')


def on_click(event):
    if event.xdata is not None:
        x = event.xdata
        clicked_datetime = mdates.num2date(event.xdata)
        #print(f"Clicked at datetime: {clicked_datetime}")
    return clicked_datetime
       # stored_datetime = clicked_datetime 
        #plt.disconnect(binding_id)
        