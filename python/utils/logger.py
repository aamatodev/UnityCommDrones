"""

This class provides the basics for logging the environment information into wandb.

"""
from torch.utils.tensorboard import SummaryWriter

from python.utils.args import Args
import logging
import wandb

class Logger:
    def __init__(self, args: Args):
        logging.basicConfig(level=logging.DEBUG)

        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            monitor_gym=True,
            save_code=True,
        )

        # Set up the writer
        self.writer = SummaryWriter(f"runs/{args.run_name}")

        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    def log(self, name, value, step):
        self.writer.add_scalar(name, value, step)
        logging.info(f"{name}: {value} at step {step}")
