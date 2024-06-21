import numpy as np
import json
import os
from tqdm import tqdm
import pandas as pd
import torch
import shutil
import copy

from baseline import *

import sympy as sp

use_cuda = torch.cuda.is_available()
gpu_id = 0
if use_cuda:
    device = torch.device('cuda', gpu_id)
else:
    device = torch.device('cpu')
print("using {}".format(device))

# dataset_config = [
#             n_samples,
#             t.tolist(),
#             input_length,
#             y0,
#             beta,
#             gamma,
#             is_scale,
#             seed,
#         ]
#         dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()

ODE_ID_DICT = {
    "Lotka_Volterra": [1, 2],
    "Lorenz": [1, 2, 3],
    "SIR": [1, 2, 3],
}

ODE_DIM_DICT = {
    "Lotka_Volterra": 2,
    "Lorenz": 3,
    "SIR": 3,
}

VARIABLE_IPAD_DICT = {
    1: "x",
    2: "y",
    3: "z",
}

VARIABLE_META_DICT = {
    "Lotka_Volterra": [r'$x$', r'$y$'],
    "Lorenz": [r'$x$', r'$y$', r'$z$'],
    "SIR": [r'$S$', r'$I$', r'$R$'],
}

META_NAME_DICT = {
    "Lotka_Volterra": "LotkaVolterraDataset",
    "Lorenz": "LorenzDataset",
    "SIR": "SIREpidemicDataset",
}

LIST_ODE = ["Lotka_Volterra"]  # ["Lotka_Volterra", "Lorenz", "SIR"]
LIST_SETTING = ["default_0"]  # ["default_0", "default_11", "default_12", "default_13"]
LIST_NOISE_RATIO = ["0.00", "0.05", "0.10", "0.15", "0.20"]# ["0.00"]  # ["0.00", "0.05", "0.10", "0.15", "0.20"]
LIST_SEED = list(range(3))#list(range(1))  # list(range(20))


def rmdirs(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print("The specified path does not exist.")
        return

    # Walk through all subdirectories and files in the folder
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)  # Remove each file
            # print(f"Removed file: {file_path}")
        for name in dirs:
            dir_path = os.path.join(root, name)
            os.rmdir(dir_path)  # Remove each directory after clearing its files
            # print(f"[finished] Removed directory: {dir_path}")

    # Optionally, remove the root folder at the end
    os.rmdir(folder_path)
    # print(f"Removed directory: {folder_path}")


def copy_file(src_path, dst_path):
    """
    Copies a file from src_path to dst_path.

    :param src_path: str - The path to the source file.
    :param dst_path: str - The path where the file should be copied.
    """
    try:
        shutil.copy2(src_path, dst_path)  # copy2 also copies metadata
        # print(f"[finished] File copied successfully from {src_path} to {dst_path}")
    except Exception as e:
        print(f"Error: {e}")


def unit_convert(input_folder, output_folder, ode_name, num_env=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # for env_id in range(num_env):

    # read_path = f"{input_folder}/{ode_name}_train_{'save'}.csv"
    read_param_path = f"{input_folder}/{ode_name}_info.json"
    write_path = f"{output_folder}/{ode_name}_{'save'}.pt"
    write_param_path = f"{output_folder}/{ode_name}_info.json"

    # data = pd.read_csv(read_path)
    state_names = VARIABLE_META_DICT[ode_name]
    state_dim = ODE_DIM_DICT[ode_name]
    input_length = 500

    read_path_0 = f"{input_folder}/{ode_name}_train_{0}.csv"
    data_0 = pd.read_csv(read_path_0)
    t = torch.tensor(data_0["t"].to_numpy())

    y_numpy = np.zeros([num_env, len(t), state_dim])
    # for idx in range(y_numpy.shape[0]):
    #     for i_var in range(state_dim):
    #         y_numpy[idx, :, i_var] = data[f"{VARIABLE_IPAD_DICT[i_var + 1]}"].to_numpy()

    dy_numpy = np.zeros([num_env, len(t), state_dim])
    # for idx in range(dy_numpy.shape[0]):
    #     for i_var in range(state_dim):
    #         dy_numpy[idx, :, i_var] = data[f"d{VARIABLE_IPAD_DICT[i_var + 1]}"].to_numpy()

    for idx in range(num_env):
        read_path = f"{input_folder}/{ode_name}_train_{idx}.csv"
        data = pd.read_csv(read_path)
        for i_var in range(state_dim):
            y_numpy[idx, :, i_var] = data[f"{VARIABLE_IPAD_DICT[i_var + 1]}"].to_numpy()
            dy_numpy[idx, :, i_var] = data[f"d{VARIABLE_IPAD_DICT[i_var + 1]}"].to_numpy()
    y = torch.tensor(y_numpy)
    dy = dict()
    dy["smooth"] = torch.tensor(dy_numpy)
    max_for_scaling = torch.tensor(np.asarray([1.0, 1.0]))
    phy_params = None

    with open(write_path, "wb") as f:
        all_var = [
            state_names,
            state_dim,
            input_length,
            t,
            y,
            dy,
            max_for_scaling,
            phy_params,
        ]
        torch.save(all_var, f)
    with open(read_param_path, "r") as f:
        param = json.load(f)
    param["n"] = len(t)
    param["t"] = list(data_0["t"].to_numpy())
    with open(write_param_path, "w") as f:
        json.dump(param, f, indent=4)




def one_time_convert_ipad_to_meta(input_folder):
    pbar = tqdm(total=(len(LIST_SETTING) * len(LIST_NOISE_RATIO) * len(LIST_ODE) * len(LIST_SEED)))
    # print("starting...")
    for one_ode in LIST_ODE:
        index_file_path = f"{input_folder}/{one_ode}_index/meta_{one_ode}_dict.json"
        with open(index_file_path, "r") as f:
            index_json = json.load(f)
        # print(index_json)
        for one_setting in LIST_SETTING:
            for one_noise_ratio in LIST_NOISE_RATIO:
                # for one_ode_id in ODE_ID_DICT[one_ode]:
                one_ode_id = 1
                for one_seed in LIST_SEED:
                    time_string = index_json["data"][one_setting][one_noise_ratio][str(one_ode_id)][str(one_seed)]
                    # print(time_string)
                    unit_convert(f"{input_folder}/{one_ode}/{time_string}/", f"{input_folder}/{one_ode}_meta/{one_setting}/{one_noise_ratio}/{one_seed}/", one_ode, num_env=5)

                    pbar.set_description(f"Processing {time_string}")
                    pbar.update(1)
    pass


def load_dataset(folder_path, ode_name, num_env=5):
    k = 5
    data_kfold = []
    input_length = 500
    is_scale = True
    param_json_path = f"{folder_path}/{ode_name}_info.json"
    with open(param_json_path, "r") as f:
        param_json = json.load(f)

    t = np.asarray(param_json["t"])
    n_samples = int(param_json["n"])
    root = "./tmp"


    if os.path.exists(root):
        rmdirs(root)
    if not os.path.exists(root):
        os.makedirs(root)



    assert ode_name in ODE_DIM_DICT
    seed = 0
    if ode_name == "Lotka_Volterra":
        # y0_id = [(1000, 2000), (10, 20)]
        # y0_ood = [(100, 200), (10, 20)]
        # alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
        # beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
        # gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
        # delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)

        # y0_id = [param_json["y0"][str(i)] for i in range(num_env)]
        y0_ood = y0_id = [0.0, 0.0]
        alpha_id = alpha_ood = [param_json["params"][str(i)][0] for i in range(num_env)]
        beta_id = beta_ood = [param_json["params"][str(i)][1] for i in range(num_env)]
        gamma_id = gamma_ood = [param_json["params"][str(i)][2] for i in range(num_env)]
        delta_id = delta_ood = [param_json["params"][str(i)][3] for i in range(num_env)]

        dataset_config = [
            n_samples,
            t.tolist(),
            input_length,
            y0_id,
            alpha_id,
            beta_id,
            gamma_id,
            delta_id,
            is_scale,
            seed,
        ]
        # print("##### Fake hash #####")
        # print(json.dumps(dataset_config, indent=4))
        dataset_config_hash = sha1(json.dumps(dataset_config).encode()).hexdigest()
        src_path_hash = f"{folder_path}/{ode_name}_{'save'}.pt"
        dst_path_hash = f"{root}/{META_NAME_DICT[ode_name]}_{dataset_config_hash}.pt"

        copy_file(src_path_hash, dst_path_hash)

        train_data = LotkaVolterraDataset(
            int(param_json["n"]),
            t,
            input_length=input_length,
            y0=y0_id,
            alpha=alpha_id,
            beta=beta_id,
            gamma=gamma_id,
            delta=delta_id,
            seed=seed,
            is_scale=is_scale,
            root=root,
            reload=False,
        )

        id_test_data = LotkaVolterraDataset(
            int(param_json["n"]),
            t,
            input_length=input_length,
            y0=y0_id,
            alpha=alpha_id,
            beta=beta_id,
            gamma=gamma_id,
            delta=delta_id,
            seed=seed,
            is_scale=is_scale,
            root=root,
            reload=False,
        )

        ood_test_data = LotkaVolterraDataset(
            int(param_json["n"]),
            t,
            input_length=input_length,
            y0=y0_id,
            alpha=alpha_id,
            beta=beta_id,
            gamma=gamma_id,
            delta=delta_id,
            seed=seed,
            is_scale=is_scale,
            root=root,
            reload=False,
        )
        data_kfold.append((train_data, id_test_data, ood_test_data))


    else:
        raise NotImplementedError
    return data_kfold, param_json["log_truth_ode_list"]



    # k = 5
    # data_kfold = []
    # T = 10
    # nT = 10 * T
    # t = np.linspace(0, T, nT)
    # input_length = int(nT // input_length_factor)
    # is_scale = True
    # for seed in range(k):
    #     np.random.seed(seed)  # Set seed
    #     if datatype == 1:
    #         # 1. Single dynamical system parameter across all training curves
    #         # 2. OOD on initial conditions
    #         y0_id = [(1000, 2000), (10, 20)]
    #         y0_ood = [(100, 200), (10, 20)]
    #         alpha_id = alpha_ood = 0.1 * 12
    #         beta_id = beta_ood = 0.005 * 12
    #         gamma_id = gamma_ood = 0.04 * 12
    #         delta_id = delta_ood = 0.00004 * 12
    #
    #     elif datatype == 2:
    #         # 1. Multiple dynamical system parameter across training curves
    #         # 2. OOD on initial conditions
    #         y0_id = [(1000, 2000), (10, 20)]
    #         y0_ood = [(100, 200), (10, 20)]
    #         alpha_id = alpha_ood = (0.1 * 12, 0.2 * 12)
    #         beta_id = beta_ood = (0.005 * 12, 0.01 * 12)
    #         gamma_id = gamma_ood = (0.04 * 12, 0.08 * 12)
    #         delta_id = delta_ood = (0.00004 * 12, 0.00008 * 12)
    #
    #     elif datatype == 3:
    #         # 1. Multiple dynamical system parameter across training curves
    #         # 2. OOD on initial conditions & dynamical system parameters
    #         y0_id = [(1000, 2000), (10, 20)]
    #         y0_ood = [(100, 200), (10, 20)]
    #
    #         alpha_id = (0.1 * 12, 0.2 * 12)
    #         beta_id = (0.005 * 12, 0.01 * 12)
    #         gamma_id = (0.04 * 12, 0.08 * 12)
    #         delta_id = (0.00004 * 12, 0.00008 * 12)
    #         alpha_ood = (0.2 * 12, 0.3 * 12)
    #         beta_ood = (0.01 * 12, 0.015 * 12)
    #         gamma_ood = (0.08 * 12, 0.12 * 12)
    #         delta_ood = (0.00008 * 12, 0.00012 * 12)
    #
    #     train_data = cls(
    #         int(n_samples * 0.8),
    #         t,
    #         input_length=input_length,
    #         y0=y0_id,
    #         alpha=alpha_id,
    #         beta=beta_id,
    #         gamma=gamma_id,
    #         delta=delta_id,
    #         seed=seed,
    #         is_scale=is_scale,
    #         root=root,
    #         reload=reload,
    #     )
    #
    #     id_test_data = cls(
    #         int(n_samples * 0.2),
    #         t,
    #         input_length=input_length,
    #         y0=y0_id,
    #         alpha=alpha_id,
    #         beta=beta_id,
    #         gamma=gamma_id,
    #         delta=delta_id,
    #         seed=seed,
    #         is_scale=is_scale,
    #         max_for_scaling=train_data.max_for_scaling,
    #         root=root,
    #         reload=reload,
    #     )
    #
    #     ood_test_data = cls(
    #         int(n_samples * 0.2),
    #         t,
    #         input_length=input_length,
    #         y0=y0_ood,
    #         alpha=alpha_ood,
    #         beta=beta_ood,
    #         gamma=gamma_ood,
    #         delta=delta_ood,
    #         seed=seed,
    #         root=root,
    #         is_scale=is_scale,
    #         max_for_scaling=train_data.max_for_scaling,
    #         reload=reload,
    #     )
    #     data_kfold.append((train_data, id_test_data, ood_test_data))


def unit_run(data_kfold, fig_save_root, truth_ode_list):
    train_data, id_test_data, ood_test_data = data_kfold[0]

    params = {
        "state_dim": train_data.state_dim,
        "is_round": True,
        "diff_method": "smooth",
        "polynomial_power": 3,  # int(args["--polynomial_power"]),

        # Fit params
        "n_inner_epochs": 10000,  # 10000
        "lr": 1e-2,  # float(args["--lr"]),
        "lambda_vrex": 1e-2,  # float(args["--lambda_vrex"]),
        "lambda_phi": 0.00,  # float(args["--lambda_phi"]),
        "n_batches": 1,

        # Test params
        "n_test_inner_epochs": 10000,  # 10000,
        "test_lr": 1e-3,
        "plot": "show",

        "filename": f"results/metaphysica_{train_data.__class__.__name__}",
    }

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # train_data.plot()
    # test_data = {"id": id_test_data, "ood": ood_test_data}
    # model, train_results, valid_results, test_results = run(train_data, test_data, params, None)
    #
    # print(f"Test ID NRMSE: {test_results['id']['nrmse']:.4f}")
    # print(f"Test OOD NRMSE: {test_results['ood']['nrmse']:.4f}")


    # trainIdx, validIdx = project_utils.getRandomSplit(len(train_data), [80, 20])
    # valid_data = SubsetStar(train_data, validIdx)
    # train_data = SubsetStar(train_data, trainIdx)

    train_results = None
    if "fit" not in params or params["fit"]:
        model = MetaPhysiCa(**params).to(device)
        train_results = model.fit(train_data, **params)

    test_data = {"train": train_data}

    # print(f"train_results: ", json.dumps(train_results, indent=4))
    # # model = torch.load(params["filename"] + ".pkl").to(device)
    # xi = model.xi.cpu().detach().numpy()
    # print("model.xi:", model.xi)
    # print("model.xi.shape:", model.xi.shape)
    # W = model.W.cpu().detach().numpy()
    # print("model.W:", model.W)
    # print("model.W.shape:", model.W.shape)
    # allW = model.allW.cpu().detach().numpy()
    # print("model.allW:", model.allW)
    # print("model.allW.shape:", model.allW.shape)
    # rhs_list = model.print()
    #
    # print("xi:", xi)
    # print("W:", W)
    # print("allW:", allW)
    # print("rhs_list:", rhs_list)

    test_results = {}
    # for key, test_data_ in test_data.items():
    #     test_results[key] = model.test(
    #         test_data_, test_filename=params["filename"] + f"_test={key}", **params
    #     )
    for key, test_data_ in test_data.items():
        test_results[key] = model.test(
            test_data_, test_filename=params["filename"] + f"_test={key}", fig_save_root=fig_save_root, truth_ode_list=truth_ode_list, **params
        )

    # formatted_equations = formula_format(xi, rhs_list)
    # for i, eq in enumerate(formatted_equations):
    #     print(f"########## x{i}' = {eq}")


def one_time_run_from_loading(input_folder):
    # args = docopt(__doc__)
    # data = args["--data"]
    # datatype = int(args["--datatype"])

    pbar = tqdm(total=(len(LIST_SETTING) * len(LIST_NOISE_RATIO) * len(LIST_ODE) * len(LIST_SEED)))
    # print("starting...")
    for one_ode in LIST_ODE:
        for one_setting in LIST_SETTING:
            for one_noise_ratio in LIST_NOISE_RATIO:
                for one_seed in LIST_SEED:
                    folder_path = f"{input_folder}/{one_ode}_meta/{one_setting}/{one_noise_ratio}/{one_seed}/"
                    fig_folder_path = f"{input_folder}/{one_ode}_fig/{one_setting}/{one_noise_ratio}/{one_seed}/"
                    data_kfold, log_truth_ode_list = load_dataset(folder_path, one_ode)
                    unit_run(data_kfold, fig_folder_path, log_truth_ode_list)

                    pbar.set_description(f"Processing: ode={one_ode},setting={one_setting},noise_ratio={one_noise_ratio},seed={one_seed}")
                    pbar.update(1)


    # if data == "lotka_volterra":
    #     data_kfold = LotkaVolterraDataset.get_standard_dataset(root='./data', datatype=datatype, n_samples=100) # 1000
    # elif data == "sir":
    #     data_kfold = SIREpidemicDataset.get_standard_dataset(root='./data', datatype=datatype, n_samples=1000)
    # else:
    #     raise NotImplementedError

    # Running on a single CV fold.
    # train_data, id_test_data, ood_test_data = data_kfold[0]
    #
    # params = {
    #     "state_dim": train_data.state_dim,
    #     "is_round": True,
    #     "diff_method": "smooth",
    #     "polynomial_power": int(args["--polynomial_power"]),
    #
    #     # Fit params
    #     "n_inner_epochs": 100,  # 10000
    #     "lr": float(args["--lr"]),
    #     "lambda_vrex": float(args["--lambda_vrex"]),
    #     "lambda_phi": float(args["--lambda_phi"]),
    #     "n_batches": 1,
    #
    #     # Test params
    #     "n_test_inner_epochs": 100,  # 10000,
    #     "test_lr": 1e-3,
    #     "plot": "show",
    #
    #     "filename": f"results/metaphysica_{train_data.__class__.__name__}",
    # }
    #
    # if not os.path.exists("./results"):
    #     os.makedirs("./results")
    # if not os.path.exists("./data"):
    #     os.makedirs("./data")
    #
    # # train_data.plot()
    # test_data = {"id": id_test_data, "ood": ood_test_data}
    # # model, train_results, valid_results, test_results = run(train_data, test_data, params, None)
    # #
    # # print(f"Test ID NRMSE: {test_results['id']['nrmse']:.4f}")
    # # print(f"Test OOD NRMSE: {test_results['ood']['nrmse']:.4f}")
    #
    # trainIdx, validIdx = project_utils.getRandomSplit(len(train_data), [80, 20])
    # print("cp 1")
    # valid_data = SubsetStar(train_data, validIdx)
    # print("cp 2")
    # train_data = SubsetStar(train_data, trainIdx)
    # print("cp 3")
    #
    # train_results = None
    # if "fit" not in params or params["fit"]:
    #     model = MetaPhysiCa(**params).to(device)
    #     train_results = model.fit(train_data, **params)
    #
    # model = torch.load(params["filename"] + ".pkl").to(device)
    # print("model.xi:", model.xi)
    # xi = model.xi.cpu().detach().numpy()
    # print("model.xi.shape:", model.xi.shape)
    # rhs_list = model.print()
    # print("xi:", xi)
    # print("rhs_list:", rhs_list)
    #
    # formatted_equations = formula_format(xi, rhs_list)
    # for i, eq in enumerate(formatted_equations):
    #     print(f"########## x{i}' = {eq}")


    # print("Validation data")
    # valid_results = model.test(valid_data, **{**params, "plot": False})

    # print("Test data: In-distribution and Out-of-distribution")
    # # Enze: Test Part
    # test_results = {}
    # for key, test_data_ in test_data.items():
    #     test_results[key] = model.test(
    #         test_data_, test_filename=params["filename"] + f"_test={key}", **params
    #     )


    pass


if __name__ == "__main__":
    # python test_build_meta_dataset.py --data=lotka_volterra --datatype=2
    one_time_convert_ipad_to_meta("ipad_dataset/")
    one_time_run_from_loading("ipad_dataset/")
    pass
