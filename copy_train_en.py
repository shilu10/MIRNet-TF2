import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import * 
from mirnet import get_enchancement_model
import argparse
from utils import charbonnier_loss, CharBonnierLoss, psnr_enchancement, PSNR
from dataloaders import LOLDataLoader
from custom_trainer import Trainer


parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--loss_function', type=str, default="charbonnier")
parser.add_argument('--n_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--checkpoint_filepath', type=str, default="checkpoint/saved/enchancement/")
parser.add_argument('--num_rrg', type=int, default=3)
parser.add_argument('--num_mrb', type=int, default=2)
parser.add_argument('--num_channels', type=int, default=64)
parser.add_argument('--summary', type=bool, default=False)
parser.add_argument('--store_model_summary', type=bool, default=False)

args = parser.parse_args()

def train():
    dataloader = LOLDataLoader("lol")
    dataloader.initialize()
    train_ds = dataloader.get_dataset(
                    subset="train",
                    batch_size=args.batch_size,
                    transform=True
                )

    val_ds = dataloader.get_dataset(
                    subset="val",
                    batch_size=args.batch_size,
                    transform=False
                )

    model = get_enchancement_model(
            num_rrg=args.num_rrg,
            num_mrb=args.num_mrb,
            num_channels=args.num_channels
        )

    if args.summary:
        model.summary()

    if args.store_model_summary:
        tf.keras.utils.plot_model(to_file="mirnet_enchancement.png")

    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    

    if args.loss_function == "charbonnier":
        loss_func = charbonnier_loss

    if args.loss_function == "l1":
        loss_func = tf.keras.metrics.MeanAbsoluteError()

    else:
        loss_func = tf.keras.metrics.MeanSquaredError()

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model, epoch=tf.Variable(1))
    manager = tf.train.CheckpointManager(
        checkpoint, directory="saved/zerodce_new", max_to_keep=5)
    status = checkpoint.restore(manager.latest_checkpoint)

    trainer = Trainer(model, loss_func, psnr_enchancement, optimizer, checkpoint, manager, 1)
    trainer.train(train_ds, val_ds)

if __name__ == '__main__':
    train()

