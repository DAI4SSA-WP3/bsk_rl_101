import argparse
import logging
import numpy as np
import time
import torch
import random
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from multi_action_ppoModule_diverse_behavior import Agent
from bsk_rl import sats, act, obs, scene, data
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import walker_delta_args
from bsk_rl.utils.orbital import random_orbit
from bsk_rl import GeneralSatelliteTasking
from bsk_rl import comm
import setproctitle
import pdb

# Set up argument parsing
parser = argparse.ArgumentParser(
    description="Run multi-satellite environment with default resources.")

# Process and environment parameters
parser.add_argument("--random_seed", type=int, default=1,
                    help="Random seed for reproducibility.")
parser.add_argument("--gpu_id", type=int, default=0,
                    help="GPU ID to use for CUDA.")
parser.add_argument("--n_satellites", type=int,
                    default=2, help="Number of satellites.")
parser.add_argument("--uniform_targets", type=int,
                    default=100, help="Number of uniform targets.")
parser.add_argument("--n_act_image", type=int, default=2,
                    help="Number of action images.")
parser.add_argument("--n_obs_image", type=int, default=2,
                    help="Number of observation images.")
parser.add_argument("--orbit_num", type=int, default=4,
                    help="Number of orbits.")
parser.add_argument("--battery_capacity", type=int, default=400,
                    help="Battery capacity size of the satellite (Wh).")
parser.add_argument("--init_battery_level", type=float, default=100.0,
                    help="Initial stored battery level in percentage")
parser.add_argument("--memory_size", type=int, default=500000,
                    help="Memory size of the satellite (Mbyte).")
parser.add_argument("--init_memory_percent", type=float, default=0.0,
                    help="Inital memory free space in percentage")
parser.add_argument("--baud_rate", type=int, default=4.3,
                    help="control baud rate of S-Band sattellite comm. in Mbps")
parser.add_argument("--randomize_enabled", type=bool, default=False,
                    help="Randomize initial stored battery, memory, disturbance, reaction wheel")
parser.add_argument("--instr_baud_rate", type=int, default=500,
                    help="control baud rate of sattellite scanning instrument")
parser.add_argument("--randomness_flag", action="store_true",
                    help="Enable randomness of environment")

# PPO parameters
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size for training.")
parser.add_argument("--gamma", type=float, default=0.99,
                    help="Discount factor for PPO.")
parser.add_argument("--learning_rate", type=float,
                    default=0.0001, help="Learning rate for PPO.")
parser.add_argument("--epochs", type=int, default=5,
                    help="Number of epochs for training.")
parser.add_argument("--train_interval", type=int, default=2,
                    help="Interval for training steps.")
parser.add_argument("--max_steps", type=int, default=1000,
                    help="Number of max training steps.")
parser.add_argument("--entropy_coef", type=float, default=0.0,
                    help="Number of training episodes.")
parser.add_argument("--diversity_coef", type=float, default=0.0,
                    help="Number of training episodes.")
parser.add_argument("--ppo_clip", type=float, default=0.2,
                    help="ppo clip parameter.")

args = parser.parse_args()


class ImagingSatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speeds_fraction")
        ),
        obs.Eclipse(norm=5700),
        obs.OpportunityProperties(
            dict(prop="priority"),
            dict(prop="opportunity_open", norm=5700.0),
            n_ahead_observe=args.n_obs_image,
        ),
        obs.OpportunityProperties(
            dict(prop="opportunity_open", norm=5700),
            dict(prop="opportunity_close", norm=5700),
            type="ground_station",
            n_ahead_observe=1,
        ),
        obs.Time(),
    ]
    action_spec = [
        act.Image(n_ahead_image=args.n_act_image),
        act.Charge(duration=20.0),
        act.Downlink(duration=20.0),
        act.Desat(duration=20.0)
    ]
    fsw_type = fsw.SteeringImagerFSWModel
    dyn_type = dyn.ManyGroundStationFullFeaturedDynModel


def train_agent(seed_num):
    # Set manual seed for reproducibility
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    np.random.seed(seed_num)
    random.seed(seed_num)

    logging.basicConfig(level=logging.WARNING)

    # Set up device
    device = torch.device(
        f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    satellite_names = []
    for i in range(args.n_satellites):
        satellite_names.append(f"Satellite{i}")

    action_names = {}
    for i in range(args.n_act_image):
        action_names[i] = f"Image_Target_{i}"

    action_names[i + 1] = "Charge"
    action_names[i + 2] = "Downlink"
    action_names[i + 3] = "Desaturate"

    inclination = -37.0  # degrees, fixed for all satellites
    altitude = 500  # km, fixed for all satellites
    eccentricity = 0  # Circular orbit
    LAN = 0  # Longitude of Ascending Node (Omega), fixed for all
    arg_periapsis = 0  # Argument of Periapsis (omega), fixed for all
    # offset = 225
    true_anomaly_offsets = [
        10-i*0.0001 for i in range(len(satellite_names))]  # degrees

    orbit_ls = []
    for offset in true_anomaly_offsets:
        orbit = random_orbit(
            i=inclination, alt=altitude, e=eccentricity, Omega=LAN, omega=arg_periapsis, f=offset
        )
        orbit_ls.append(orbit)
    multiSat = []
    # Battery sizes in Wh, converted to Joules
    max_battery_cap = 400
    min_battery_cap = 200
    battery_sizes = np.linspace(min_battery_cap, max_battery_cap, num=len(
        satellite_names)) * 3600

    # Define the satellite arguments
    sat_args_list = []
    index = 0
    for orbit, battery_size in zip(orbit_ls, battery_sizes):
        sat_args = dict(
            u_max=0.4,
            omega_max=0.1,
            servo_Ki=5.0,
            servo_P=150,
            dataStorageCapacity=args.memory_size * 8e6,
            storageInit=int(args.memory_size *
                            args.init_memory_percent/100) * 8e6 if not args.randomize_enabled else np.random.uniform(args.memory_size * 3600 * 0.2, args.memory_size * 3600 * 0.8),
            instrumentBaudRate=args.instr_baud_rate * 1e6,  # 1Mbps
            transmitterBaudRate=-1*args.baud_rate * 1e6,
            batteryStorageCapacity=args.battery_capacity * 3600,
            storedCharge_Init=int(args.battery_capacity *
                                  args.init_battery_level / 100) * 3600 if not args.randomize_enabled else np.random.uniform(args.battery_capacity * 3600 * 0.2, args.battery_capacity * 3600 * 0.8),
            panelArea=1.0,
            panelEfficiency=20.0,
            basePowerDraw=-10.0,
            instrumentPowerDraw=-30.0,
            transmitterPowerDraw=-25.0,
            thrusterPowerDraw=-80.0,
            imageAttErrorRequirement=0.1,
            imageRateErrorRequirement=0.1,
            disturbance_vector=np.array(
                [0.0, 0.0, 0.0]) if not args.randomize_enabled else lambda: np.random.normal(scale=0.0001, size=3),
            maxWheelSpeed=6000.0,
            wheelSpeeds=np.array(
                [0.0, 0.0, 0.0]) if not args.randomize_enabled else lambda: np.random.uniform(-3000, 3000, 3),
            desatAttitude="nadir",
            oe=orbit
        )
        sat = ImagingSatellite(f"EO-{index}", sat_args)
        multiSat.append(sat)
        index += 1

    duration = args.orbit_num * 5700.0

    target_total = args.uniform_targets
    env = GeneralSatelliteTasking(
        satellites=multiSat,
        # scenario=scene.CityTargets(target_total),
        scenario=scene.UserDefinedTargets(target_total),
        # rewarder=data.UniqueImageReward(),
        rewarder=data.MultipleImageReward(),
        time_limit=duration,
        communicator=comm.LOSCommunication(),
        log_level="WARNING",
        terminate_on_time_limit=True,
        failure_penalty=-100.0,
        # activate this line to record visualization
        # vizard_dir="./tmp/vizard",
        # activate this line to record visualization
        # vizard_settings=dict(showLocationLabels=1),
    )

    # Generate a random integer for uniqueness
    random_suffix = random.randint(1000, 9999)

    # Incorporate the random integer into the log_dir path
    log_dir = (f'results_test/Default/'
               f'{args.n_satellites}Sat_{args.uniform_targets}'
               f'{seed_num}seed_'
               f'{args.orbit_num}orbit_'
               f'{args.memory_size}mem_'
               f'{args.init_memory_percent}initMemoryDiv_'
               f'{args.baud_rate}baudRateInMem_'
               f'{args.instr_baud_rate}baudRateInstr_'
               f'{args.battery_capacity}BattCap_'
               f'{args.init_battery_level}initBatt_'
               f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_{random_suffix}'
               )
    if not args.randomness_flag:
        log_dir += "_no_env_randomness"
    writer = SummaryWriter(log_dir=log_dir)

    env.reset()

    obs_space = env.observation_space[0].shape[0]
    act_space = env.action_space[0].n
    ppoAgent = Agent(
        state_dim=obs_space * args.n_satellites,
        # Multi-discrete action space dimensions for each satellite
        action_dims=[act_space] * args.n_satellites,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        policy_clip=args.ppo_clip,
        gamma=args.gamma,
        lamda=0.99,
        adam_lr=args.learning_rate,
        device=device,
        entropy_coef=args.entropy_coef,
        diversity_coef=args.diversity_coef
    )
    score_history = []
    n_steps = 0
    i_episode = 0

    while n_steps < args.max_steps:

        if args.randomness_flag:
            current_state, info = env.reset()
        else:
            current_state, info = env.reset(seed=seed_num)
        terminated, truncated, done = False, False, False
        score = 0
        action_count = defaultdict(int)

        # Track total actions for percentage calculation
        total_actions = 0
        battery_usage = {sat: [] for sat in satellite_names}
        memory_usage = {sat: [] for sat in satellite_names}
        # Initialize the nested dictionary
        action_frequencies = {
            sat: {action: 0 for action in action_names} for sat in satellite_names}
        epsiode_length = 0
        battery_total_charge_amount = {sat: 0 for sat in satellite_names}

        while not done:
            acts, probs, val = ppoAgent.choose_action(
                np.concatenate(current_state))

            # Step the environment with multi-satellite actions
            next_state, reward, terminated, truncated, info = env.step(
                tuple(acts))
            done = 1 if (terminated or truncated) else 0
            n_steps += 1
            epsiode_length += 1
            score += reward
            # Track action counts for each satellite
            for index, sat in enumerate(satellite_names):
                action_frequencies[sat][acts[index]] += 1
            total_actions += 1

            # Record battery and memory usage for each satellite
            for i, sat in enumerate(satellite_names):
                # Track battery for satellite `sat`
                battery_usage[sat].append(current_state[i][1].item())
                # Track memory for satellite `sat`
                memory_usage[sat].append(current_state[i][0].item())

                # battery_charged_fraction = next_state[i][1].item(
                # ) - current_state[i][1].item()
                # battery_charge_amount = 0
                # if battery_charged_fraction * battery_sizes[i] > -200000:
                #     # reward += 1
                #     battery_charge_amount = battery_charged_fraction + \
                #         power_used_per_act / battery_sizes[i]
                #     # print(f"battery_charge_amount {battery_charge_amount}")
                #     battery_total_charge_amount[sat] += battery_charge_amount

            ppoAgent.store_data(current_state, acts, probs, val, reward, done)

            current_state = next_state

        if i_episode % args.train_interval == 0:
            losses = ppoAgent.learn()
            # Log losses to TensorBoard
            writer.add_scalar("Loss/Actor_Loss",
                              losses["actor_loss"], i_episode)
            writer.add_scalar("Loss/Critic_Loss",
                              losses["critic_loss"], i_episode)
            writer.add_scalar("Loss/Entropy", losses["entropy"], i_episode)
            writer.add_scalar("Loss/Diversity_Loss",
                              losses["diversity_loss"], i_episode)
            writer.add_scalar("Loss/Total_Loss",
                              losses["total_loss"], i_episode)

        # Append score and calculate averages
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        print(f'action taken in this episode: {epsiode_length}')
        print(f'epsiode length / terminate: {next_state[0][-1]}')

        # Log episode reward, average score
        writer.add_scalar('Episode Reward', score, n_steps)
        writer.add_scalar('Average Reward (last 10)', avg_score, n_steps)
        i_episode += 1
        # pdb.set_trace()
        # Log mean battery and memory usage for each satellite
        for sat in satellite_names:
            mean_battery_usage = 1 - \
                np.mean(battery_usage[sat]) if battery_usage[sat] else 0
            mean_memory_usage = np.mean(
                memory_usage[sat]) if memory_usage[sat] else 0

            writer.add_scalar(f'{sat}/Mean Battery Usage',
                              mean_battery_usage, i_episode)
            writer.add_scalar(f'{sat}/Mean Memory Usage',
                              mean_memory_usage, i_episode)

        # Log battery charge / action usage percentage for each satellite
        for sat in satellite_names:
            # Total actions for this satellite
            sat_total_actions = sum(action_frequencies[sat].values())
            for action, count in action_frequencies[sat].items():
                action_percentage = (count / sat_total_actions) * \
                    100 if sat_total_actions > 0 else 0
                writer.add_scalar(
                    f'{sat}/Action {action_names[action]} Usage Percentage', action_percentage, i_episode
                )

            writer.add_scalar(
                f'{sat}/Battery Total Charged Amount',
                battery_total_charge_amount[sat], i_episode
            )
            print(
                f'{sat}/Battery Total Charged Amount: {battery_total_charge_amount[sat]}')

        # Print action counts for each satellite
        print(f'Episode {i_episode} - Action Counts per Satellite:')
        for sat in satellite_names:
            print(f'  {sat}:')
            for action, count in action_frequencies[sat].items():
                print(f'    {action}: {count} times')

        print(f'Episode {i_episode} - Avg Score: {avg_score:.1f}')

    writer.close()


if __name__ == "__main__":
    setproctitle.setproctitle("CentralizedPPO-default-BSK")
    for seeds in range(args.random_seed):
        print("Running training for seed: ", seeds)
        train_agent(seed_num=seeds)
