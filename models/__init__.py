from ray.rllib.models import ModelCatalog

from models.serverconv import ServerConvNetwork

ModelCatalog.register_custom_model("serverconv", ServerConvNetwork)