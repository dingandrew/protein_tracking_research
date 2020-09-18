class Track:
    def __init__(self, locs, id, centroid, state, origin, conf):
        self.__locs = locs # immutable
        self.id = id
        self.__centroid = centroid  # immutable
        self.state = state
        self.origin = origin
        self.confidence = conf

    @property
    def locs(self):
        return self.__locs

    @property
    def centroid(self):
        return self.__centroid

    def __repr__(self):
        return "\n<Track \nlocs:%s \nid:%s \ncentroid:%s \nstate:%s \norigin:%s \nconf:%s>" % (len(self.locs), self.id, self.centroid, self.state, self.origin, self.confidence)

    def __eq__(self, other):
        if not isinstance(other, Track):
            # don't attempt to compare against unrelated types
            return NotImplemented
        comp = self.locs == other.locs
        if type(comp) is not bool:
            comp = comp.all()
        return comp and (self.centroid == other.centroid)

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        # NOTE: only use immutable members
        key = (len(self.locs), self.centroid[0],
               self.centroid[1], self.centroid[2])
        return hash(key)
