import os
import json
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
sys.path.append('../PyTorch_CIFAR10')

from cifar10_models.densenet import densenet121, densenet161
CHECKPOINT_PATH = './saved_models/'

# CIFAR-10 related constants
NUM_CLASSES = 10
LABEL_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
MODELS = ['resnet20', 'resnet32', 'vgg16_bn', 'vgg19_bn'] #, 'densenet121', 'densenet161']

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8
BATCH_SIZE=128

# Mean and Std from CIFAR-10
NORM_MEAN = np.array([0.491, 0.482, 0.446])
NORM_STD = np.array([0.247, 0.243, 0.261])

PATCH_SIZE = 3

def get_label_index(lab_str):
    assert lab_str in LABEL_NAMES, f"Label \"{lab_str}\" not found. Check the spelling of the class."
    return LABEL_NAMES.index(lab_str)

# No resizing and center crop necessary as images are already preprocessed.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN,
                         std=NORM_STD)
])


# Function to calculate accuracy
def calculate_accuracy(net, dataloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def vanilla_model_test(net):
    # Load CIFAR-10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    # Ensure the model is in evaluation mode
    net.eval() 
    # Evaluate on the training data
    train_accuracy = calculate_accuracy(net, trainloader, device)
    print('Accuracy of the network on the training images: %d %%' % train_accuracy)

    # Evaluate on the test data
    test_accuracy = calculate_accuracy(net, testloader, device)
    print('Accuracy of the network on the test images: %d %%' % test_accuracy)

def place_patch(img, patch):
    for i in range(img.shape[0]):
        # h_offset = np.random.randint(0,img.shape[2]-patch.shape[1]-1)
        # w_offset = np.random.randint(0,img.shape[3]-patch.shape[2]-1)
        h_offset = int(0.33 * (img.shape[2] - patch.shape[1]))
        w_offset = int(0.33 * (img.shape[3] - patch.shape[2]))
        TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]
        img[i,:,h_offset:h_offset+patch.shape[1],w_offset:w_offset+patch.shape[2]] = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return img

def evaluate(net, patch, testloader):
    # Apply the patch to see if the network classifies correctly
    net.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for img, labels in tqdm(testloader, leave=False):
            img = place_patch(img, patch)
            img, labels = img.to(device), labels.to(device)
            pred = net(img)
            correct += (pred.argmax(dim=-1) == labels).sum()
            total += labels.shape[0]
    return ("%.3f" % (correct.item() / total))

def eval_patch(net, patch, testloader, target_class):
    net.eval()
    tp, tp_5, total = 0, 0, 0
    with torch.no_grad():
        for img, labels in tqdm(testloader, leave=False):
            img = place_patch(img, patch)
            img, labels = img.to(device), labels.to(device)
            pred = net(img)
            tp += torch.logical_and(pred.argmax(dim=-1) == target_class, labels != target_class).sum()
            tp_5 += torch.logical_and((pred.topk(5, dim=-1)[1] == target_class).any(dim=-1), labels != target_class).sum()
            total += (labels != target_class).sum()
    return ("%.3f" % (tp.item() / total.item())), ("%.3f" % (tp_5.item() / total.item()))

def eval_patch_untargeted(net, patch, testloader):
    net.eval()
    tp, tp_5, total = 0, 0, 0
    with torch.no_grad():
        for img, labels in tqdm(testloader, leave=False):
            img = place_patch(img, patch)
            img, labels = img.to(device), labels.to(device)
            pred = net(img)
            tp += (pred.argmax(dim=-1) != labels).sum()
            tp_5 += (~pred.topk(5, dim=-1)[1].eq(labels.unsqueeze(1))).all(dim=1).sum()
            total += labels.shape[0]
    return ("%.3f" % (tp.item() / total)), ("%.3f" % (tp_5.item() / total))

def train_patch_untargeted(net, trainloader, patch_size, num_epochs):
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.01)
    loss_module = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        t = tqdm(trainloader, leave=False)
        for img, labels in t:
            img = place_patch(img, patch)
            img, labels = img.to(device), labels.to(device)
            pred = net(img)
            # Calculate the loss for untargeted attack
            loss = -loss_module(pred, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f'Epoch {epoch+1}/{num_epochs}')
        print (f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    return patch


def train_patch(net, trainloader, target_class, patch_size, num_epochs):
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.01)
    loss_module = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        t = tqdm(trainloader, leave=False)
        for img, _ in t:
            img = place_patch(img, patch)
            img = img.to(device)
            pred = net(img)
            labels = torch.zeros(img.shape[0], device=pred.device, dtype=torch.long).fill_(target_class)
            loss = loss_module(pred, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f'Epoch {epoch+1}/{num_epochs}')
        print (f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
    return patch

def main(args):
    if args.non_targeted:
        UntargetedAttack(args)
    else:
        TargetedAttack(args)

def UntargetedAttack(args):
    CHECKPOINT_PATH = './saved_models/non_targeted/'
    model_name = args.model
    patch_size_list = [3, 5, 7, 9, 16]
    if model_name == "densenet121":
        print (f"Model {model_name}...")
        # net = DenseNet.from_pretrained("densenet121")
        net = densenet121(pretrained=True)
        # net.model.load_state_dict(torch.load(f"densenet121.pt"))
    elif model_name == "densenet161":
        print (f"Model {model_name}...")
        net = densenet161(pretrained=True)
        # net.model.load_state_dict(torch.load(f"densenet161.pt"))
    else:
        print(f"Model: {model_name}. Target")
        net = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_name}", pretrained=True)
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if not os.path.exists(f"{CHECKPOINT_PATH}{model_name}"):
        os.makedirs(f"{CHECKPOINT_PATH}{model_name}")
    net.to(device)
    net.eval()
    # Train the patch
    results_dict = {}
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    for patch_size in patch_size_list:
        save_patch_name = f"{CHECKPOINT_PATH}{model_name}/{patch_size}.pt"
        save_str = "patch_" + str(patch_size)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
        testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
        if os.path.exists(save_patch_name):
            print(f"Patch {save_str} already exists. Skipping...")
            # results = evaluate(net, torch.load(save_patch_name), testloader)
            results = eval_patch_untargeted(net, torch.load(save_patch_name), testloader)
            print (results)
            results_dict[save_str] = results
            continue
        patch = train_patch_untargeted(net, trainloader, patch_size=patch_size, num_epochs=args.num_epochs)
        # Evaluate the patch
        # results = evaluate(net, patch, testloader)
        results = eval_patch_untargeted(net, patch, testloader)
        print (results)
        results_dict[save_str] = results
        # Save this patch 
        torch.save(patch.data, f"{CHECKPOINT_PATH}{model_name}/{patch_size}.pt")
    # Save the results_dict to a file
    with open(f"{CHECKPOINT_PATH}{model_name}_results.json", "w") as f:
        json.dump(results_dict, f)
    print(f"Results saved to {CHECKPOINT_PATH}{model_name}_results.json")

def TargetedAttack(args):
    model_name = args.model
    patch_size_list = [3, 5, 7, 9, 16]
    if model_name == "densenet121":
        print (f"Model {model_name}...")
        # net = DenseNet.from_pretrained("densenet121")
        net = densenet121(pretrained=True)
        # net.model.load_state_dict(torch.load(f"densenet121.pt"))
    elif model_name == "densenet161":
        print (f"Model {model_name}...")
        net = densenet161(pretrained=True)
        # net.model.load_state_dict(torch.load(f"densenet161.pt"))
    else:
        print(f"Model: {model_name}. Target")
        net = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_name}", pretrained=True)
    # check if the directory exists
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    if not os.path.exists(f"{CHECKPOINT_PATH}{model_name}"):
        os.makedirs(f"{CHECKPOINT_PATH}{model_name}")
    net.to(device)
    net.eval()
    # Train the patch
    results_dict = {}
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    for patch_size in patch_size_list:
        for target_class in range(NUM_CLASSES):
        # Load CIFAR-10 training and test datasets
            save_patch_name = f"{CHECKPOINT_PATH}{model_name}/{patch_size}_{target_class}.pt"
            save_str = "patch_" + str(patch_size) + "_target_" + str(target_class)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
            testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
            if os.path.exists(save_patch_name):
                print(f"Patch {save_str} already exists. Skipping...")
                # results = evaluate(net, torch.load(save_patch_name), testloader)
                results = eval_patch(net, torch.load(save_patch_name), testloader, target_class)
                results_dict[save_str] = results
                print (results)
                continue
            print(f"Target Class: {LABEL_NAMES[target_class]}")
            patch = train_patch(net, trainloader, target_class, patch_size=patch_size, num_epochs=args.num_epochs)
            # Evaluate the patch
            # results = evaluate(net, patch, testloader)
            results = eval_patch(net, patch, testloader, target_class)
            print (results)
            results_dict[save_str] = results
            # Save this patch 
            torch.save(patch.data, f"{CHECKPOINT_PATH}{model_name}/{patch_size}_{target_class}.pt")
    # Save the results_dict to a file
    # print (results_dict)
    # results_json = {model_name: {
    #     patch_size: [{item : results_dict[item]["results"]} for item in range(NUM_CLASSES)],
    # }}
    with open(f"{CHECKPOINT_PATH}{model_name}_results.json", "w") as f:
        json.dump(results_dict, f)
    print(f"Results saved to {CHECKPOINT_PATH}{model_name}_results.json")

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--patch_size", type=int, default=3, help="Size of the patch")
    argparse.add_argument("--target_class", type=int, default=0, help="Target class")
    argparse.add_argument("--model", type=str, default="resnet20", help="Model to use")
    argparse.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train the patch")
    argparse.add_argument("--non_targeted", action="store_true", default=False, help="Train the patch for non-targeted attack")
    args = argparse.parse_args()
    main(args)