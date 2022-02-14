from representations_config import TranscribeConfig
from new_extract_representations import extract
import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)#, config_path=None)

@hydra.main(config_name="config", config_path=None)
def hydra_main(cfg: TranscribeConfig):
    extract(cfg, layer=cfg.layer, destination_path=cfg.destination, csv_file=cfg.csv_file)


if __name__ == '__main__':
    hydra_main()