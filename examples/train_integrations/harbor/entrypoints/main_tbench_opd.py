"""
Main entrypoint for on-policy distillation training on Harbor tasks.
"""
import signal
import sys

import ray
from loguru import logger

from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from examples.train.on_policy_distillation.main_on_policy_distill import OnPolicyDistillationTrainer
from .main_harbor import HarborExp, HarborSkyRLConfig, HARBOR_DEFAULT_CONFIG, _deep_merge

import yaml


class OnPolicyDistillationHarborExp(HarborExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = OnPolicyDistillationHarborExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    initialize_ray(cfg)

    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM on head node, shutting down Ray...")
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        logger.info("Shutting down Ray on head node...")
        ray.shutdown()


if __name__ == "__main__":
    main()
