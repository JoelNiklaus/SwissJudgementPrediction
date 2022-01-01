class ResultCell:
    def __init__(self, mean=0, std=0, min=0, connector='±', show_min=False, empty=False):
        self.mean = mean
        self.std = std
        self.min = min
        self.connector = connector
        self.show_min = show_min
        self.empty = empty  # we got now result
        self.num_decimals = 1

    def round(self, num):
        """Convert a result from 0-1 to a number between 0 and 100 rounded to 2 decimals"""
        return (num * 100).round(self.num_decimals)

    def __str__(self):
        if self.empty:
            return "–"
        mean_std = f"{self.round(self.mean)} {{\small {self.connector} {self.round(self.std)}}}"
        return mean_std + (f" ({self.round(self.min)})" if self.show_min else "")
