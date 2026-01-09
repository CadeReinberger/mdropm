
def compute_area_shoelace():
    return .5 * np.abs(sum(x*np.roll(y, 1) - y*np.roll(x, 1)))


