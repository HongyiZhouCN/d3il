import logging
import multiprocessing as mp
import os
import random

import numpy as np
import torch
import wandb
from envs.gym_stacking_env.gym_stacking.envs.stacking import CubeStacking_Env

from agents.utils.sim_path import sim_framework_path
from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Stacking_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            max_steps_per_episode: int = 500
    ):
        super().__init__(seed, device, render, n_cores)

        self.n_contexts = n_contexts
        self.n_trajectories_per_context = n_trajectories_per_context

        self.max_steps_per_episode = max_steps_per_episode

        self.test_contexts = np.load(sim_framework_path("environments/dataset/data/stacking/test_contexts.pkl"),
                                     allow_pickle=True)

        # pre-define the mode
        self.mode_1 = {'r': 0, 'g': 1, 'b': 2}
        self.mode_2 = {'rg': 0, 'rb': 1, 'gr': 2, 'gb': 3, 'br': 4, 'bg': 5}
        self.mode_3 = {'rgb': 0, 'rbg': 1, 'grb': 2, 'gbr': 3, 'brg': 4, 'bgr': 5}

        self.modes = np.load(sim_framework_path("environments/dataset/data/stacking/mode_prob.pkl"),
                             allow_pickle=True)

        self.mode_encoding_3 = np.zeros(6)
        self.mode_encoding_2 = np.zeros(6)

        for m_key in self.mode_3.keys():

            self.mode_encoding_3[self.mode_3[m_key]] = self.modes[m_key]
            self.mode_encoding_2[self.mode_3[m_key]] = self.modes[m_key]

        self.mode_encoding_1 = np.array([self.mode_encoding_3[i] + self.mode_encoding_3[i+1] for i in [0, 2, 4]])

        self.mode_encoding_1 = torch.tensor(self.mode_encoding_1)
        self.mode_encoding_2 = torch.tensor(self.mode_encoding_2)
        self.mode_encoding_3 = torch.tensor(self.mode_encoding_3)

    def eval_agent(self, agent, contexts, context_ind, mode_encoding, mode_encoding_1_box, mode_encoding_2_box,
                   successes, successes_1, successes_2, cpu_set, pid, context_id_dict={}):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        env = CubeStacking_Env(max_steps_per_episode=self.max_steps_per_episode, render=self.render)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)
        print(f'core {cpu_set} proceeds Context {contexts} with Rollout context_ind {context_ind}')

        for i, context in enumerate(contexts):

            sim_step = 0

            agent.reset()

            obs = env.reset(random=False, context=self.test_contexts[context])

            pred_action, _, _ = env.robot_state()
            pred_action = pred_action.astype(np.float32)

            done = False
            while not done:

                obs = np.concatenate((pred_action, obs))

                pred_action = agent.predict(obs)[0]
                pred_action[:7] = pred_action[:7] + obs[:7]

                obs, reward, done, info = env.step(pred_action)
                sim_step += 1

            ctxt_id = context_id_dict[context]

            if len(info['mode']) > 2:
                mode_encoding_1_box[ctxt_id, context_ind[i]] = torch.tensor(self.mode_1[info['mode'][:1]])
                mode_encoding_2_box[ctxt_id, context_ind[i]] = torch.tensor(self.mode_2[info['mode'][:2]])

                mode_encoding[ctxt_id, context_ind[i]] = torch.tensor(self.mode_3[info['mode'][:3]])

            elif len(info['mode']) > 1:
                mode_encoding_1_box[ctxt_id, context_ind[i]] = torch.tensor(self.mode_1[info['mode'][:1]])
                mode_encoding_2_box[ctxt_id, context_ind[i]] = torch.tensor(self.mode_2[info['mode'][:2]])

            elif len(info['mode']) > 0:
                mode_encoding_1_box[ctxt_id, context_ind[i]] = torch.tensor(self.mode_1[info['mode'][:1]])

            else:
                pass

            successes[ctxt_id, context_ind[i]] = torch.tensor(info['success'])
            successes_1[ctxt_id, context_ind[i]] = torch.tensor(info['success_1'])
            successes_2[ctxt_id, context_ind[i]] = torch.tensor(info['success_2'])

    def cal_KL(self, mode_encoding, successes, prior_encoding, n_mode=6):

        mode_probs = torch.zeros([self.n_contexts, n_mode])

        if successes.sum() == 0:
            return 0, 0

        for c in range(self.n_contexts):

            for num in range(n_mode):

                mode_probs[c, num] = torch.tensor([sum(mode_encoding[c, successes[c, :] == 1] == num)
                                                   / self.n_trajectories_per_context])

        mode_probs /= (mode_probs.sum(1).reshape(-1, 1) + 1e-12)

        mode_probs = mode_probs[torch.nonzero(mode_probs.sum(1), as_tuple=True)[0]]

        entropy = - (mode_probs * torch.log(mode_probs + 1e-12) / torch.log(
            torch.tensor(n_mode))).sum(1).mean()

        log_ = (mode_probs * torch.log(prior_encoding + 1e-12) / torch.log(
            torch.tensor(n_mode))).sum(1).mean()

        KL = - entropy - log_

        return entropy, KL

    ################################
    # we use multi-process for the simulation
    # n_contexts: the number of different contexts of environment
    # n_trajectories_per_context: test each context for n times, this is mostly used for multi-modal data
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent, cpu_cores=None):

        log.info('Starting trained model evaluation')

        mode_encoding_1_box = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        mode_encoding_2_box = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        # mode encoding for 2 and 3 boxes are the same
        mode_encoding = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()

        successes = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        successes_1 = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        successes_2 = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()


        self.n_cores = len(cpu_cores) if cpu_cores is not None else 10

        contexts = np.random.randint(0, 60, self.n_contexts) if self.n_contexts != 60 else np.arange(60)
        context_idx_dict = {c: i for i, c in enumerate(contexts)}

        contexts = np.repeat(contexts, self.n_trajectories_per_context)

        context_ind = np.arange(self.n_trajectories_per_context)
        context_ind = np.tile(context_ind, self.n_contexts)

        repeat_nums = (self.n_contexts * self.n_trajectories_per_context) // self.n_cores
        repeat_res = (self.n_contexts * self.n_trajectories_per_context) % self.n_cores

        workload_array = np.ones([self.n_cores], dtype=int)
        workload_array[:repeat_res] += repeat_nums
        workload_array[repeat_res:] = repeat_nums

        assert np.sum(workload_array) == len(contexts)

        ind_workload = np.cumsum(workload_array)
        ind_workload = np.concatenate(([0], ind_workload))


        ctx = mp.get_context('spawn')

        p_list = []
        if self.n_cores > 1:
            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "agent": agent,
                        "contexts": contexts[ind_workload[i]:ind_workload[i + 1]],
                        "context_ind": context_ind[ind_workload[i]:ind_workload[i + 1]],
                        "mode_encoding": mode_encoding,
                        "mode_encoding_1_box": mode_encoding_1_box,
                        "mode_encoding_2_box": mode_encoding_2_box,
                        "successes": successes,
                        "successes_1": successes_1,
                        "successes_2": successes_2,
                        "pid": i,
                        "cpu_set": set([int(cpu_cores[i])]),
                        "context_id_dict": context_idx_dict
                    },
                )
                # print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, contexts, context_ind, mode_encoding, mode_encoding_1_box, mode_encoding_2_box,
                            successes, successes_1, successes_2, pid=0, cpu_set=set([0]), context_id_dict=context_idx_dict)

        box1_success_rate = torch.mean(successes_1).item()
        box2_success_rate = torch.mean(successes_2).item()

        success_rate = torch.mean(successes).item()

        entropy_1, KL_1 = self.cal_KL(mode_encoding_1_box, successes_1, self.mode_encoding_1, n_mode=3)
        entropy_2, KL_2 = self.cal_KL(mode_encoding_2_box, successes_2, self.mode_encoding_2, n_mode=6)
        entropy_3, KL_3 = self.cal_KL(mode_encoding, successes, self.mode_encoding_3, n_mode=6)

        return {'box_1_success_rate':box1_success_rate, 'box_2_success_rate':box2_success_rate, 'box_3_success_rate':success_rate,
                'entropy_1':entropy_1, 'KL_1':KL_1, 'entropy_2':entropy_2, 'KL_2':KL_2, 'entropy_3':entropy_3, 'KL_3':KL_3}