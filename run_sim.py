import os
import logging

import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import torch
from agents.utils.sim_path import sim_framework_path


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="avoiding_config.yaml")
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)
    
    # TODO: insert agent.load_pretrained_model() here with relative path
    path = "/home/hongyi/Codes/demo_acc_rl/d3il/logs/avoiding/sweeps/beso/2024-03-11/16-06-09/agent_name=beso,agents.num_sampling_steps=4,agents.sigma_max=1,agents.sigma_min=0.1,agents=beso_agent,seed=0,window_size=1"
    sv_name = "eval_best_beso.pth"
    agent.load_pretrained_model(path, sv_name=sv_name)

    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()