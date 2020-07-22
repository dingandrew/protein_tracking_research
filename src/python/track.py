class Track:
    def __init__(self, locs, id, centroid, state, origin, mask):
        self.locs = locs
        self.id = id
        self.centroid = centroid
        self.state = state
        self.origin = origin
        self.mask = mask.copy()

    def __repr__(self):
        return "<Track \nlocs:%s \nid:%s \ncentroid:%s \nstate:%s \norigin:%s>" % (len(self.locs), self.id, self.centroid, self.state, self.origin)
