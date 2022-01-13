import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter

import model

class ModelTrainer():
    def __init__(self, optimizer, lossfn, model: model.TuxDriverModel, training_dataloader: DataLoader, test_dataloader: DataLoader):
        self.optimizer = optimizer
        self.lossfn = lossfn
        self.model = model
        self.training_dataloader = training_dataloader
        self.test_dataloader = test_dataloader
        self.writer = SummaryWriter()

    def _epoch_test(self):
        self.model.eval()
        size = len(self.test_dataloader.dataset)
        loss_total = 0

        for batch, (X, y) in enumerate(self.test_dataloader):
            pred = self.model(X)
            loss = self.lossfn(pred, y).item()
            loss_total += loss

            current = batch * len(X)
            self.writer.add_scalar("Loss/Test/Batch", loss, batch)
            print("\rTesting... ", current, "/", size, end="")

        print("\nTest avg: ", loss_total / len(self.test_dataloader))

        self.model.train()
        return loss_total / len(self.test_dataloader)


    def _epoch_train(self):
        size = len(self.training_dataloader.dataset)
        total_loss = 0

        for batch, (X, y) in enumerate(self.training_dataloader):
            # Compute prediction and loss
            pred = self.model(X)
            loss = self.lossfn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
            loss, current = loss.item(), batch * len(X)
            print(f"\rloss: {loss:>7f}  [{current:>5d}/{size:>5d}]", end="")

            self.writer.add_scalar("Loss/Train/Batch", loss, batch)

            total_loss += loss
        
        print("\nAvg loss: ", total_loss / len(self.training_dataloader))
        return total_loss / len(self.training_dataloader)


    def train(self, epochs):
        for n in range(epochs):
            print("\n########### Epoch number " + str(n) + " ###########")
            avg_loss_train = self._epoch_train()
            avg_loss_test = self._epoch_test()
            self.writer.add_scalar("Loss/Train/Epochs", avg_loss_train, n)
            self.writer.add_scalar("Loss/Test/Epochs", avg_loss_test, n)
            self.writer.flush()
        
        self.writer.close()
    
    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

def setup_pytorch(interop, intraop):
    torch.set_num_interop_threads(interop)
    torch.set_num_threads(intraop)
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-folder", type=str, help="Set the folder of the training dataset", default="./dataset/training")
    parser.add_argument("--testing-folder", type=str, help="Set the folder of the testing dataset", default="./dataset/testing")
    parser.add_argument("--learning-rate", type=float, help="Set the learning rate for training", required=True)
    parser.add_argument("--batch-size", type=int, help="Set batch size", default=64)
    parser.add_argument("--epochs", type=int, help="Set number of epochs", default=50)
    parser.add_argument("--output-file", type=str, help="Set the filename of the model", required=True)
    parser.add_argument("--intra-op-threads", type=int, help="Set pytorch intraop threads")
    parser.add_argument("--inter-op-threads", type=int, help="Set pytorch interop threads")
    parser.add_argument("--import-weights", type=str, help="Import starting weights from a pre-trained model", required=False)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_pytorch(args.inter_op_threads, args.intra_op_threads)
    print("intra op: " + str(torch.get_num_threads()))
    print("inter op: " + str(torch.get_num_interop_threads()))

    neural_network = model.TuxDriverModel()
    if args.import_weights != None:
        neural_network.load_state_dict(torch.load(args.import_weights))
    training_dataloader = DataLoader(model.TuxDriverDataset(args.training_folder, transform=ToTensor()), batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_dataloader = DataLoader(model.TuxDriverDataset(args.testing_folder, transform=ToTensor()), batch_size=args.batch_size, shuffle=True, num_workers=1)
    # optimizer = optim.SGD(neural_network.parameters(), args.learning_rate, momentum=0.9)
    optimizer = optim.Adam(neural_network.parameters())
    lossfn = nn.BCELoss()
    trainer = ModelTrainer(optimizer, lossfn, neural_network, training_dataloader, test_dataloader)

    try:
        trainer.train(args.epochs)
    except KeyboardInterrupt:
        pass

    trainer.save_model(args.output_file)
