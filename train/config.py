import json
import os


def build_flag(model_name, **kwargs):
    keys = kwargs.keys()
    for key in keys:
        model_name += "_" + key + str(kwargs[key])
    return model_name


class BaseConfig(object):
    def __init__(self):
        self.model_flag = build_flag("model")
        self.device = "cuda"

    def add_param(self,
                  param: str,
                  default_value):
        if param.startswith("_") or param.startswith("__"):
            raise ValueError("The param name '{}' which starting with '_' or '__' is not usable for Config objects."
                             "It should be added by '__setattr__(param, default_value)' method.")
        self.__setattr__(param, default_value)

    def del_param(self,
                  param: str):
        self.__delattr__(param)

    def get_params(self):
        return self.__dict__

    def save(self, path):
        config_json = json.dumps(self.get_params(), indent=4)
        with open(os.path.join(path, "config.json"), "w") as file:
            file.write(config_json)

    def load(self, path):
        with open(path, "r") as file:
            config_json = json.load(file)
        for p in config_json.keys():
            self.add_param(p, config_json[p])
        return self

