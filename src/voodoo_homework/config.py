import yaml


def load_config():
    with open("config.yml", 'r') as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    return data
