import json


def ReadConfiguration(config_name='config.json'):
    with open('./env/scenarios/scenario_one/configs/' + config_name,
              'r',
              encoding='UTF-8') as f:
        return json.load(f)
