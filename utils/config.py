import yaml

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            elif isinstance(value, list) and all(isinstance(v, dict) for v in value):
                setattr(self, key, [Config(v) for v in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                if all(isinstance(v, Config) for v in value):
                    result[key] = [v.to_dict() for v in value]
                else:
                    result[key] = value
            else:
                result[key] = value
        return result

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return Config(config_dict)