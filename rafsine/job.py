class ConstantArrival:
    def __init__(self, load, duration):
        self.load = load
        self.duration = duration
    def step(self):
        return (self.load, self.duration)