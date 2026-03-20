"""
Main entrypoint for on-policy distillation with teacher logits on Harbor tasks.

Combines the HarborGenerator (Harbor/Daytona agent environment) with the
DistillationTrainer (vLLM-based teacher scoring for top-K logprobs).

The teacher model is served via a separate vLLM engine (supports AWQ/GPTQ quantization)
and provides top-K log-probability distributions for student-generated sequences.
"""

import signal
import sys

import torch
import ray
import yaml
from loguru import logger

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    register_policy_loss,
    reduce_loss,
    masked_mean,
)
from skyrl.backends.skyrl_train.distillation_trainer import DistillationTrainer
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from .main_harbor import HarborExp, HarborSkyRLConfig, HARBOR_DEFAULT_CONFIG, _deep_merge


class OnPolicyDistillationLogitsHarborTrainer(DistillationTrainer):
    """
    On-policy distillation trainer with teacher logits for Harbor tasks.

    Uses teacher top-K logprobs as the reward signal, replacing environment
    rewards with KL divergence between teacher and student.
    """

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Compute KL-based reward from teacher/ref logprobs."""
        loss_mask = data["loss_mask"]
        teacher_action_log_probs = data["base_action_log_probs"]
        action_log_probs = data["action_log_probs"]

        # Reverse KL as reward: -(student_logprobs - teacher_logprobs)
        rewards = -(action_log_probs - teacher_action_log_probs) * loss_mask
        data["rewards"] = rewards

        kl_mean = masked_mean(rewards.abs(), loss_mask, dim=-1).mean().item()
        self.all_metrics.update({"distill/token_kl_mean": kl_mean})

        return data


# Register custom advantage estimator and policy loss for distillation
@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    return token_level_rewards, token_level_rewards


@register_policy_loss("importance_sampling")
def compute_importance_sampling_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    loss = -torch.exp(log_probs - old_log_probs) * advantages
    loss = reduce_loss(loss, loss_mask, "seq_mean_token_sum_norm", config.max_seq_len)
    return loss, 0.0


class OnPolicyDistillationLogitsHarborExp(HarborExp):
    """Harbor experiment with on-policy distillation + teacher logits."""

    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationLogitsHarborTrainer(*args, **kwargs)

    def _setup_trainer(self):
        """Override to create teacher vLLM inference engines."""
        trainer = super()._setup_trainer()

        # Create teacher engines if configured
        # NOTE: create_teacher_inference_engines_from_config must be ported to
        # skyrl.train.entrypoints.main_base from the old skyrl-train tree.
        if hasattr(self.cfg, "teacher") and self.cfg.teacher.model_path is not None:
            from skyrl.train.entrypoints.main_base import create_teacher_inference_engines_from_config

            teacher_engines, teacher_tokenizer = create_teacher_inference_engines_from_config(
                self.cfg, self.tokenizer
            )
            trainer.setup_teacher_engine(
                teacher_engines,
                student_tokenizer=self.tokenizer,
                teacher_tokenizer=teacher_tokenizer,
            )
            logger.info(f"Teacher engine created for {self.cfg.teacher.model_path}")
        else:
            logger.warning(
                "No teacher.model_path configured. Running without teacher logits. "
                "Set teacher.model_path to enable teacher scoring."
            )

        return trainer


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg):
    exp = OnPolicyDistillationLogitsHarborExp(cfg)
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
