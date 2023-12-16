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
# import Image to use
from PIL import Image

sys.path.append('../PyTorch_CIFAR10')

# CIFAR-10 related constants
NUM_CLASSES = 10
LABEL_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 8
BATCH_SIZE=128

# Mean and Std from CIFAR-10
NORM_MEAN = np.array([0.491, 0.482, 0.446])
NORM_STD = np.array([0.247, 0.243, 0.261])

# No resizing and center crop necessary as images are already preprocessed.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN,
                         std=NORM_STD)
])

# generate 0.1 to 1.0 with 0.1 step
placement_offset = 0.1 * np.arange(1, 10, 1)
print (placement_offset)

# (w, h)
results_patch_placement = [(0.4, 0.4), (0.6, 0.2), (0.2, 0.5), (0.1, 0.8)]
names = ["car1", "bird", "car2", "truck"]

def place_patch(ori_img, patch, h_placement, w_placement):
    img = ori_img.clone()
    for i in range(img.shape[0]):
        h_offset = int(h_placement * (img.shape[2] - patch.shape[1]))
        w_offset = int(w_placement * (img.shape[3] - patch.shape[2]))
        TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]
        img[i,:,h_offset:h_offset+patch.shape[1],w_offset:w_offset+patch.shape[2]] = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return img

def show_prediction(img, label, pred, name=None, K=5):
    
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    img = img * NORM_STD + NORM_MEAN
    img = np.clip(img, 0, 1)
    
    # Plot on the left the image with the true label as title.
    # On the right, have a horizontal bar plot with the top k predictions including probabilities
    
    fig, ax = plt.subplots(1, 2, figsize=(10,2), gridspec_kw={'width_ratios': [1, 1]})
    
    ax[0].imshow(img)
    ax[0].set_title(LABEL_NAMES[label])
    ax[0].axis('off')
    
    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    topk_vals = topk_vals[0]
    topk_idx = topk_idx[0]
    # plot topk_vals as horizontal bar
    ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', color=["C0" if topk_idx[i]!=label else "C2" for i in range(K)])

    # ax[-1].barh(np.arange(K), topk_vals*100.0, align='center', color=["C0" if topk_idx[i]!=label else "C2" for i in range(K)])
    ax[-1].set_yticks(np.arange(K))
    ax[-1].set_yticklabels([LABEL_NAMES[c] for c in topk_idx])
    ax[-1].invert_yaxis()
    ax[-1].set_xlabel('Confidence')
    ax[-1].set_title('Predictions')

    if name != None:
        plt.savefig('patched_img_with_results_{}.png'.format(name))
    plt.show()
    plt.close()

def plot_img(img, pred):
    # plot the img as well as the prediction bars
    # sort the pred bars and only show first 5 in %
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    img = img * NORM_STD + NORM_MEAN
    img = np.clip(img, 0, 1)
    pred = pred.detach().cpu().numpy()
    pred = pred.squeeze(0)
    pred = pred.argsort()[-5:][::-1]
    pred = pred / pred.sum()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.barh(np.arange(5), pred)
    # plt.yticks(np.arange(10), LABEL_NAMES)
    plt.tight_layout()
    plt.show()


model_name = "resnet32"
net = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_name}", pretrained=True)

# Test all patch results
def targeted_eval(img, patch_size, target_class):
    results = {}
    for w_placement in placement_offset:
        for h_placement in placement_offset:
            print ("Patch Size: {}. Placement: {}/{}".format(patch_size, w_placement, h_placement))
            patch_name = "./saved_models/{}/{}_{}.pt".format(model_name, patch_size, target_class)
            patch = torch.load(patch_name)
            # place patch
            patched_img = place_patch(img, patch, h_placement=h_placement, w_placement=w_placement)
            # predict the patched image
            net.eval()
            with torch.no_grad():
                logits = net(patched_img)
                pred = logits.argmax(dim=1)
                # print ("Predicted Results: {}".format(pred[0].item()))
                # print ("Predicted Name: {}".format(LABEL_NAMES[pred[0].item()]))
                show_prediction(patched_img, true_class, logits, K=5)
                results[(w_placement, h_placement)] = pred[0].item()
    print (results)
    return results

def untargeted_eval(img, patch_size):
    results = {}
    for w_placement in placement_offset:
        for h_placement in placement_offset:
            print ("Patch Size: {}. Placement: {}/{}".format(patch_size, w_placement, h_placement))
            patch_name = "./saved_models/fixed_location/non_targeted/{}/{}.pt".format(model_name, patch_size)
            patch = torch.load(patch_name)
            # place patch
            patched_img = place_patch(img, patch, h_placement=h_placement, w_placement=w_placement)
            # predict the patched image
            net.eval()
            with torch.no_grad():
                logits = net(patched_img)
                pred = logits.argmax(dim=1)
                # print ("Predicted Results: {}".format(pred[0].item()))
                # print ("Predicted Name: {}".format(LABEL_NAMES[pred[0].item()]))
                # show_prediction(patched_img, true_class, logits, K=5)
                results[(w_placement, h_placement)] = pred[0].item()
    print (results)
    return results

def plot_results(img, results, img_name, patch_size):
    correct_color = "green"
    wrong_color = "red"
    img = img.squeeze(0)
    img = img.permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    img = img * NORM_STD + NORM_MEAN
    img = np.clip(img, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for placement in results:
        if results[placement] == true_class:
            color = correct_color
        else:
            color = wrong_color
        plt.scatter(placement[0]*img.shape[1], placement[1]*img.shape[0], color=color)
    plt.savefig('(fixed)result_{}_{}.png'.format(img_name, patch_size))
    plt.show()


def poster_plot(img):
    for placement in results_patch_placement:
        print ("Placement: {}".format(placement))
        name = names[results_patch_placement.index(placement)]
        patch_name = "./saved_models/non_targeted/{}/{}.pt".format(model_name, 5)
        patch = torch.load(patch_name)
        # place patch
        patched_img = place_patch(img, patch, h_placement=placement[1], w_placement=placement[0])
        # predict the patched image
        net.eval()
        with torch.no_grad():
            logits = net(patched_img)
            pred = logits.argmax(dim=1)
            # print ("Predicted Results: {}".format(pred[0].item()))
            print ("Predicted Name: {}".format(LABEL_NAMES[pred[0].item()]))
            show_prediction(patched_img, true_class, logits, name, K=5)
# load one picture named as resized_image_32x32.png
img_name = "duke_bus"
img = Image.open(img_name + ".png")
img = transform(img)
img = img.unsqueeze(0)

# predict the original image
net.eval()
with torch.no_grad():
    logits = net(img)
    pred = logits.argmax(dim=1)
    print(pred)
    print(LABEL_NAMES[pred[0].item()])

true_class = pred[0].item()
# poster_plot(img)
# patch_size = 5
horse_idx = 7
for patch_size in [16]:
    results = targeted_eval(img, patch_size, horse_idx)
    plot_results(img, results, img_name, patch_size)