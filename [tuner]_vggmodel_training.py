import os
import pathlib
import shutil
import torch
import torchvision
import torchvision.transforms.v2
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm
import torchinfo
import random
import matplotlib.pyplot as plt
import shutil
import tempfile
import torchtune
import ray
from ray import tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import hyperopt



TARGET_LABELS = ['逢', '謬', '雄', '醋', '餡', '陪', '跎', '隆', '阜', '辟', '鄧', '詭', '踢', '黎', '遞', '跑', '馱', '雁', '酊', '迪', '鏍', '訐', '這', '裹', '鍊', '霾', '鄰', '詔', '辨', '錳', '辯', '鯉', '霑', '首', '賤', '鄂', '賒', '適', '黨', '部', '頰', '露', '陛', '賊', '鳴', '魄', '諫', '覦', '讒', '赫']
EXPERIMENT_DIR = "/content/image_classification_experiments"
EXPERIMENT_NAME = "VGG_experiments"

def split_dataset(base_dir, train_ratio=0.7):
    """
    Splits images from class subdirectories into train/test sets.

    Args:
        base_dir (str): Path to the dataset directory (e.g., '/content/data')
        train_ratio (float): Proportion of images for training (default=0.7)
    """
    # Define paths for train and test directories
    train_dir = os.path.join(f'{base_dir}_split', 'train')
    test_dir = os.path.join(f'{base_dir}_split', 'test')

    # Create train/test directories if they don't exist
    try:
        os.makedirs(train_dir)
        os.makedirs(test_dir)
    except FileExistsError:
        return f'{base_dir}_split'

    # Supported image extensions
    img_exts = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # Process each class directory
    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)

        # Skip non-directories and our new train/test directories
        if not os.path.isdir(class_path) or class_name in ['train', 'test']:
            continue

        # Prepare destination directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Get all image files
        images = [f for f in os.listdir(class_path)
                 if f.lower().endswith(img_exts)]
        random.shuffle(images)  # Shuffle randomly

        # Split indices
        split_idx = int(len(images) * train_ratio)
        train_files = images[:split_idx]
        test_files = images[split_idx:]

        # Move files to their new locations
        for f in train_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(train_class_dir, f)
            shutil.copyfile(src, dst)

        for f in test_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(test_class_dir, f)
            shutil.copyfile(src, dst)

        # Remove original class directory if empty
        if not os.listdir(class_path):
            os.rmdir(class_path)
    return f'{base_dir}_split'

def data_deploy(image_folder: os.PathLike):
    target_folders = []
    for foldername in os.listdir(image_folder):
        if foldername in TARGET_LABELS:
            target_folders.append(os.path.join(image_folder, foldername))

    sample_folder_path = f"{image_folder}_sample"
    try:
        os.mkdir(sample_folder_path)
        for folder in target_folders:
            os.system(f"cp -r {folder} {sample_folder_path}")
        
    except FileExistsError:
        pass
    
    # Usage
    split_datadir = split_dataset(image_folder)
    return split_datadir, sample_folder_path

# Load dataset
def create_dataloaders(data_dir:str,
                       transforms_list: List,
                       val_transforms: torchvision.transforms.Compose,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, List]:
    data_train_list = []
    for data_transforms in transforms_list:
        data_train = torchvision.datasets.ImageFolder(f'{data_dir}/train', transform=data_transforms)
        data_train_list.append(data_train)
    data_train = torchtune.datasets.ConcatDataset(data_train_list)
    data_test = torchvision.datasets.ImageFolder(f'{data_dir}/test', transform=val_transforms)
    label_names = data_test.classes
    train_dataloader = DataLoader(
        data_train,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        data_test,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, label_names

class pure_background_transforms(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        # Assume img to be a torch tensor
        generated_color = random.uniform(0, 1)
        # Do not use in-place replacement!!!
        new_img = torch.where(img >= 0.99, generated_color, img)
        return new_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(random_seed={self.random_seed})"

class noise_background_transforms(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, img):
        noise_img = torch.rand(img.shape)
        new_img = torch.where(img >= 0.99, noise_img, img)
        return new_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

def transformsGenerator(zoom: bool, rotation: bool, background: str, gray_scale: bool = False):
    transforms_sequence = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=3),
    ]
    if zoom:
        transforms_sequence.append(
            torchvision.transforms.v2.RandomZoomOut(fill=1, side_range=(1.0, 1.5), p=0.5) # Why I cannot use 255 here?
        )
    
    if rotation:
        transforms_sequence.append(
            torchvision.transforms.RandomRotation(degrees=(-60, 60), fill=1)
        )
    
    if background == 'pure':
        transforms_sequence.append(
            pure_background_transforms(),
        )
    elif background == 'noise':
        transforms_sequence.append(
            noise_background_transforms(),
        )
    elif background == 'origin':
            pass
    else:
        raise ValueError(f"background must be one of 'pure', 'noise', 'origin', but got {background}")
    
    transforms_sequence.append(torchvision.transforms.Resize((224, 224))) # The image is 50*50 originally. GaussianBlur destroy image details
    
    if gray_scale:
        transforms_sequence.append(torchvision.transforms.Grayscale(num_output_channels=1))
    
    return torchvision.transforms.Compose(transforms_sequence)

class modelTrainer():
    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 device: torch.device,
                 loss_fn: nn.Module,
                 optimizer: torch.optim,
                 ):
        self.model = model
        self.best_acc = None
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_step(self, testing: bool=False) -> Tuple[float, float]:
        # Set model to train mode & move model to device
        self.model = self.model.to(self.device)
        self.model.train()
        train_loss, train_acc = 0, 0 # Epoch-level loss & acc

        # Batch-level model update
        count = 0
        for data_batch, label_batch in tqdm(self.train_dataloader, desc="Training"):
            # Move data to device
            data_batch, label_batch = data_batch.to(self.device), label_batch.to(self.device)
            # Forward pass
            label_probs = self.model(data_batch)
            # Loss calculation
            loss = self.loss_fn(label_probs, label_batch) # This will give the average of the loss for this batch, i.e., loss = sum(loss of each datapoint) / batch_size
            train_loss += loss.item()
            # zero-grad
            self.optimizer.zero_grad()
            # loss backward
            loss.backward()
            # step the model
            self.optimizer.step()
            # Evaluation: acc
            label_pred = label_probs.argmax(dim = -1)
            acc = (label_pred == label_batch).sum().item() / len(label_batch)
            train_acc += acc
            count += 1
            if testing and count > 10:
                break

        train_acc /= count # This will give the average of the loss for this epoch, i.e., loss = sum(batch-level average losss) / total_count_of_batch
        train_loss /= count
        return train_acc, train_loss

    def test_step(self, testing: bool=False):
        self.model.to(self.device)
        self.model.eval()
        test_loss, test_acc = 0, 0
        count = 0
        with torch.inference_mode():
            for data_batch, label_batch in tqdm(self.test_dataloader, desc="Testing"):
                data_batch, label_batch = data_batch.to(self.device), label_batch.to(self.device)
                label_probs = self.model(data_batch)
                loss = self.loss_fn(label_probs, label_batch)
                test_loss+=loss.item()
                label_pred = label_probs.argmax(dim=-1)
                acc = (label_pred==label_batch).sum().item() / len(label_batch)
                test_acc += acc
                count += 1
                if testing and count > 3:
                    break

        test_loss /= count
        test_acc /= count
        return test_loss, test_acc

    def train(self,
              num_epochs: int,
              start_epoch: int,
              testing: bool=False,):
        input_size = next(iter(self.train_dataloader))[0].shape
        print(torchinfo.summary(self.model, input_size = input_size)) # Level this line here to check whether there are bugs in the model architecture
        results = [] # Containing Dicts

        for epoch in range(start_epoch, num_epochs):
            print(f"Epoch: {epoch+1} | ", end="")

            train_loss, train_acc = self.train_step(testing)
            test_loss, test_acc = self.test_step(testing)

            print(f"train_loss: {train_loss:.6f} | " +
                  f"train_acc: {train_acc:.6f} | " +
                  f"test_loss: {test_loss:.6f} | " +
                  f"test_acc: {test_acc:.6f}")
            results.append({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
            })

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            with tempfile.TemporaryDirectory() as checkpoint_dir: # Open a temporary folder that can be used as context manager. On completion of the context or destruction of the temporary directory object, the newly created temporary directory and all its contents are removed. 
                cp_data_path = os.path.join(checkpoint_dir, 'data.pkl')
                with os.open(cp_data_path, "wb") as fp:
                    ray.cloudpickle.dump(checkpoint_data, fp)
                ray.tune.report(
                    metrics={'loss': test_loss, 'accuracy': test_acc},
                    checkpoint=ray.state.Checkpoint.from_directory(checkpoint_dir)
                )

        return self.best_acc, results

def plot_training_results(results, fig_save_path):
    """
    Plots training and testing loss and accuracy from the results and saves to the given path.

    Args:
        results (list of dict): List containing dictionaries with keys:
            'train_loss', 'train_acc', 'test_loss', 'test_acc'
        save_path (str): Path (including filename without extension) to save the plot image.
    """
    epochs = list(range(len(results)))
    train_loss = [entry['train_loss'] for entry in results]
    test_loss = [entry['test_loss'] for entry in results]
    train_acc = [entry['train_acc'] for entry in results]
    test_acc = [entry['test_acc'] for entry in results]

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, test_loss, 'r-', label='Test Loss')
    plt.title('Training and Testing Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy')
    plt.plot(epochs, test_acc, 'r-', label='Test Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    os.system(f"mkdir -p {fig_save_path}")
    plt.savefig(f"{fig_save_path}/loss_acc.png")
    plt.close()

class VGGmodel(nn.Module):
    def __init__(self, num_classes: int, conv_block_count, dense_layer_count, num_units, in_channels: int = 3):
        super().__init__()
        conv_block_count = conv_block_count
        dense_layer_count = dense_layer_count
        num_units = num_units
        num_classes = num_classes

        features = []
        in_channels = in_channels
        out_channels = 64
        for i in range(conv_block_count):
            features.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            features.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            features.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2

        self.features = nn.Sequential(*features)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        in_channels = in_channels * 49
        classifier = []
        for i in range(dense_layer_count):
            classifier.extend([
                nn.Linear(in_channels, num_units),
                nn.ReLU(),
                nn.Dropout(),
            ])
            in_channels = num_units
        classifier.append(nn.Linear(in_channels, num_classes)) # Last layer for classification
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1) # Do not flatten batch dim
        x = self.classifier(x)
        return x

def train_model(config):
    data_dir = config['data_dir']
    zoom = True
    rotation = True
    backgrounds = config['background_augmentation']
    gray_scale = True
    batchsize = config['batch_size']
    train_transforms = [transformsGenerator(zoom, rotation, background, gray_scale) for background in backgrounds]
    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(50, 50)),
        torchvision.transforms.Grayscale(),
    ])
    train_dataloader, test_dataloader, label_names = create_dataloaders(data_dir, train_transforms, val_transforms, batch_size=batchsize)

    conv_block_count=config['conv_block_count']
    if conv_block_count >= 5:
        raise ValueError(f"Conv_block cannot be more than 4, but got {conv_block_count}!")
    dense_layer_count=config['dense_layer_count']
    num_units=config['num_units']
    in_channels=1 # Fixed

    vgg_model = VGGmodel(len(label_names), conv_block_count, dense_layer_count, num_units, in_channels)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vgg_model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vgg_model.parameters(), lr = config['lr'])

    checkpoint = ray.train.get_checkpoint() 
    # Access the latest reported checkpoint to resume from if one exists
    # Checkpoints are saved locally to the trials's working directory by ray: ~/ray_results/...
    # ray.train.get_checkpoint() accesses only the checkpoint of the trial that calls it: no cross-trial leakage
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = f"{checkpoint_dir}/data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = ray.cloudpickle.load(fp)
            start_epoch = checkpoint_state['epoch']
            vgg_model.load_state_dict(checkpoint_state['model_state_dict'])
            optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
    else:
        start_epoch = 0

    # zoom: bool, rotation: bool, background: str, gray_scale: bool = False):
    mytrainer = modelTrainer(vgg_model, train_dataloader, test_dataloader, device, loss_fn, optimizer)
    epochs = 500 # Meaningless because max_t, just put it for running train()
    mytrainer.train(num_epochs=epochs, start_epoch=start_epoch)

def main():
    data_dir = '/content/DetectChineseCharacters'
    split_datadir, sample_datadir = data_deploy(data_dir)
    
    config = {
        'conv_block_count': tune.choice([1, 2, 3, 4]), # >5 image becomes 0
        'dense_layer_count': tune.choice([1, 2, 3, 4]),
        'num_units': tune.choice([1024, 2048, 4096, 8192]),
        'background_augmentation': tune.choice([['pure', 'origin'], ['noise', 'origin'], ['noise', 'pure', 'origin']]),
        'batch_size': tune.choice([16, 32, 64]),
        'lr': tune.loguniform(1e-6, 1e-3),
        # 'epochs': tune.choice([100]), # Meaningless because max_t = 100
        'data_dir': split_datadir,
        }
    
    scheduler = ASHAScheduler(        
            max_t=100, # Each trial should be trained with 100 times at most
            grace_period=10,
            reduction_factor=2, # The aggressiveness of trial pruning. At each halving stage, only the top 1/reduction_factor (top 50% here) of trials continue. Higher values prune more aggressively.
        )     
    
    init_searchalg = hyperopt.HyperOptSearch(metric="accuracy")
    storage_dir = os.path.join(EXPERIMENT_DIR, EXPERIMENT_NAME)
    if os.path.isfile(storage_dir) and any(pathlib.Path(storage_dir).iterdir()): # if the folder is non-empty
        init_searchalg.restore_from_dir(os.path.join(EXPERIMENT_DIR, EXPERIMENT_NAME))
            
    tuner = tune.Tuner(
            trainable=train_model,
            param_space=config, # Use either random sampling primitives to specify distribution or use grid search
            run_config=tune.RunConfig(
                name=EXPERIMENT_NAME,
                storage_path=EXPERIMENT_DIR,
            ),
            tune_config=tune.TuneConfig(
                search_alg=init_searchalg,
                num_samples=10,
                scheduler=scheduler,
                metric='accuracy',
                mode='max',
                max_concurrent_trials=4,
            ),
    )
    results = tuner.fit()
    # Add codes to capture keyboard interrupt & store algorithm state & output storage location for future resume
    # Add codes to search for any previous experiments and resume it.
    
    best_result = results.get_best_result(metric='accuracy', mode='max')
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_result.last_result['accuracy']}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        experiment_result_pth = os.path.join(EXPERIMENT_DIR, EXPERIMENT_NAME)
        zip_pth = os.path.join(EXPERIMENT_DIR, f"{EXPERIMENT_NAME}.tar")
        shutil.make_archive(zip_pth, 'tar', experiment_result_pth)
    
    