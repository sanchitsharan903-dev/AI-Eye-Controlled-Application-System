import json
import os


class Calibration:
    def __init__(self):

        # 3×3 calibration points
        self.points = [
            (100, 100),
            (640, 100),
            (1180, 100),

            (100, 360),
            (640, 360),
            (1180, 360),

            (100, 620),
            (640, 620),
            (1180, 620)
        ]

        self.current_point = 0
        self.samples = {}
        self.is_finished = False

    def reset(self):
        self.current_point = 0
        self.samples = {}
        self.is_finished = False

    def next_point(self):
        self.current_point += 1

        if self.current_point >= len(self.points):
            self.is_finished = True

    def get_current_point(self):
        if self.current_point < len(self.points):
            return self.points[self.current_point]
        return None