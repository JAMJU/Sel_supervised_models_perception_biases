import hydra
import os
import idr_torch



os.environ['WORLD_SIZE'] = str(len(idr_torch.hostnames))
os.environ['NODE_RANK'] = str(idr_torch.rank//len(idr_torch.hostnames))

os.environ['MASTER_PORT']=str(77777)




from hydra.core.config_store import ConfigStore
from hydra_configs.pytorch_lightning.callbacks import ModelCheckpointConf

from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, SGDConfig, BiDirectionalConfig, \
    UniDirectionalConfig, GCSCheckpointConfig
from deepspeech_pytorch.training import train

cs = ConfigStore.instance()
cs.store(name="config", node=DeepSpeechConfig)
cs.store(group="optim", name="sgd", node=SGDConfig)
cs.store(group="optim", name="adam", node=AdamConfig)
cs.store(group="checkpoint", name="file", node=ModelCheckpointConf)
cs.store(group="checkpoint", name="gcs", node=GCSCheckpointConfig)
cs.store(group="model", name="bidirectional", node=BiDirectionalConfig)
cs.store(group="model", name="unidirectional", node=UniDirectionalConfig)


@hydra.main(config_name="config")
def hydra_main(cfg: DeepSpeechConfig):
    train(cfg=cfg)


if __name__ == '__main__':
    hydra_main()
