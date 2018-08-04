class Line():
    def __init__(self):
        self.last_detected = False
        self.last_n_fit = []
        self.mean_x = None
        self.mean_fit = None
        self.current_fit = [np.array([False])]
        self.curvature_radius = None
        self.x_pixels = None
        slef.y_pixels = None
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

        #distance in meters of vehicle center from the line
        self.line_base_pos = None 