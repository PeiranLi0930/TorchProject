import torch
from torch import nn
from torchvision import transforms
import data_setup, engine
from helper_functions import download_data, set_seeds, plot_loss_curves
from utils import device_check
from module import StandardVit
from torchinfo import summary
from torch.optim.lr_scheduler import LambdaLR


# CONFIG
IMG_SIZE = HEIGHT = WIDTH = 224 # image info
CHANNELS = 3
PIN_MEMORY = True # avoid unnecessary copies between CPU and GPU
BATCH_SIZE = 512 # 4096 by default
PATCH_SIZE = 16
TRANSFORMER_LAYER_NUM = 12
HIDDEN_UNIT = 3072
HEAD = 12
MSA_DROPOUT = .0
MLP_DROPOUT = .1
EMBEDDING_DROPOUT = .1
BETA_1 = 0.9 # for Adam optimizer
BETA_2 = 0.999
WEIGHT_DECAY = 0.3
LEARNING_RATE = 0.008

# number of patches
N = (HEIGHT * WIDTH) // (PATCH_SIZE ** 2)


# Get data and set image_path
image_path = download_data(
    source = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    destination = "pizza_steak_sushi"
    )
train_dir = image_path / "train"
test_dir = image_path / "test"

# Create Dataloader
img_transform = transforms.Compose(
        [
                transforms.Resize([IMG_SIZE, IMG_SIZE]),
                transforms.ToTensor()
        ]
)

train_data, test_data, class_names = data_setup.create_dataloaders(
    train_dir = train_dir,
    test_dir = test_dir,
    transform = img_transform,
    batch_size = BATCH_SIZE)

# Build the model
vit = StandardVit(img_size = IMG_SIZE,
                  in_channels = CHANNELS,
                  patch_size = PATCH_SIZE,
                  transformer_layers_num = TRANSFORMER_LAYER_NUM,
                  mlp_size = HIDDEN_UNIT,
                  h = HEAD,
                  msa_dropout = MSA_DROPOUT,
                  mlp_dropout = MLP_DROPOUT,
                  embedding_dropout = EMBEDDING_DROPOUT,
                  num_classes = len(class_names))

vit = nn.DataParallel(vit)

# Print the model info
model_config = summary( model = vit, input_size = (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
                        col_names = ["input_size", "output_size", "num_params", "trainable"],
                        col_width = 20,
                        row_settings = ["var_names"])

# Train
optimizer = torch.optim.Adam(params = vit.parameters(),
                             lr = LEARNING_RATE,
                             betas = (BETA_1, BETA_2),
                             weight_decay = WEIGHT_DECAY)
num_training_steps = 10000
num_warmup_steps = 1000

# Define a lambda function for the learning rate schedule
lr_lambda = lambda step: step ** (-0.5) if step != 0 else 1e-4
scheduler = LambdaLR(optimizer, lr_lambda)

loss = nn.CrossEntropyLoss()
set_seeds()

result = engine.train(vit, train_data, test_data, optimizer, scheduler, loss, epochs = 1000,
                      device = device_check())

plot_loss_curves(result)
