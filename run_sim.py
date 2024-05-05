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
    path = "/home/hongyi/Codes/demo_acc_rl/d3il/logs/models/avoiding/ddpm_act/20-28-20/agent_name=ddpm_encdec,agents.model.n_timesteps=16,agents=ddpm_encdec,group=obstacle_avoidance_ddpm_encdec_seeds,seed=0,simulation.n_cores=30,simulation.n_trajectories=480,window_size=8"
    sv_name = "eval_best_ddpm.pth"
    agent.load_pretrained_model(path, sv_name=sv_name)

    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()