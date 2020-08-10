class Track:
    def __init__(self, locs, id, centroid, state, origin, conf):
        self.locs = locs
        self.id = id
        self.centroid = centroid
        self.state = state
        self.origin = origin
        self.confidence = conf

    def __repr__(self):
        return "\n<Track \nlocs:%s \nid:%s \ncentroid:%s \nstate:%s \norigin:%s \nconf:%s>" % (len(self.locs), self.id, self.centroid, self.state, self.origin, self.confidence)
