from learning.ccea.run import runCCEA
from learning.ppo.run import runPPO

import argparse
from pathlib import Path

if __name__ == "__main__":

    # Arg parser variables
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        default="",
        help="Experiment batch",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="",
        help="Experiment name",
        type=str,
    )
    parser.add_argument(
        "--algorithm",
        default="",
        help="Learning algorithm name",
        type=str,
    )
    parser.add_argument(
        "--domain",
        default="",
        help="Domain name",
        type=str,
    )
    parser.add_argument("--trial_id", default=0, help="Sets trial ID", type=int)

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = Path(__file__).parent

    # Set configuration folder
    batch_dir = dir_path / "experiments" / "yamls" / args["batch"]

    match(args["algorithm"]):
        case "CCEA":
            # Run learning algorithm
            runCCEA(
                batch_dir=batch_dir,
                batch_name=args["batch"],
                experiment_name=args["name"],
                trial_id=args["trial_id"],
            )
        case "PPO":
            runPPO(
                batch_dir=batch_dir,
                batch_name=args["batch"],
                experiment_name=args["name"],
                trial_id=args["trial_id"],
            )

    
