from env.scenarios.scenario_one.configs.read_config import ReadConfiguration


class Scenario():
    def __init__(self):
        self.config = ReadConfiguration()
        self.area_points = self.config['area']
