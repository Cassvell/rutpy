from pyhwm2014 import HWM14, HWM14Plot


hwm14Obj = HWM14( altlim=[90,500], altstp=1, ap=[-1, 35], day=45,
        option=1, ut=-6,  glat=19.747,  glon=-99.182 ,verbose=False, year=2020 ) 

hwm14Obj2 = HWM14( altlim=[90,500], altstp=1, ap=[-1, 35], day=45,
        option=1, ut=-6,  glat=19.747,  glon=-99.182 ,verbose=True, year=2020)

hwm14Gbj = HWM14Plot( profObj=hwm14Obj )
