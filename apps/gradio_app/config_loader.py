import yaml

def load_model_configs(config_path: str = "configs/model_ckpts.yaml") -> dict:
    with open(config_path, 'r') as f:
        return {cfg['model_id']: cfg for cfg in yaml.safe_load(f)}