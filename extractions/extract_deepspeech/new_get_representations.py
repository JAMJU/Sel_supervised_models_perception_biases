import hydra
from hydra.core.config_store import ConfigStore

from Extraction.inference_config import TranscribeConfig
from Extraction.new_extract_representations import extract
#from representations_config import DeepSpeechConfig, AdamConfig, SGDConfig, BiDirectionalConfig, \
#    UniDirectionalConfig, GCSCheckpointConfig
#hydra.initialize(config_path=None)

cs = ConfigStore.instance()
cs.store(name="config", node=TranscribeConfig)#, config_path=None)

@hydra.main(config_name="config", config_path=None)
def hydra_main(cfg: TranscribeConfig):
    extract(cfg=cfg, layer=cfg.layer, destination_path = cfg.destination, list_file = cfg.list_file, fmri = cfg.fmri) # conv1, conv2, rnn1, ..., rnn?, fully_connected


if __name__ == '__main__':
    hydra_main()

# python new_get_representations.py model.model_path=/gpfswork/rech/tub/uzz69cv/checkpoints_deepspeech/english/epch_20_step_153551_english.ckpt
# audio_path=/gpfswork/rech/tub/uzz69cv/wav_to_transform/wordvowels16000/
# destination=/gpfsscratch/rech/tub/uzz69cv/transfo_multi/deepspeech_english
# layer=conv1
