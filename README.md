# adversarial-patch
PyTorch implementation of adversarial patch 
python==3.8.18 pytorch==2.0.0

Run attack:

- `python make_patch.py --cuda --netClassifier resnet18 --max_count 100 --image_size 32 --patch_type square --outf log`


# TODO
1. pretrained model Resnet18, Densenet121, vgg16, in -- cifar10
3. Evaluate the effect of patch size. For example, generate patches with the size 3x3, 5x5, 7x7, 16x16
4. untargeted attack
5. Use different parameters, such as the number of iterations, the learning rate, the optimizer.
6. Adding Vision Transformer as a baseline model
7. Find an entity out of the dataset to utilize our patch attack