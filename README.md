# Ants-challenge

Ants Challenge on Crowd.ai

https://www.crowdai.org/challenges/ants-challenge-part-1#overview

# Setup

## Torch

Total ants = 72

## Idea

1. Divide the image into patches of 96*96
2. For each patch its class is denoted by the ant with 'X' mark nearest to the center of the patch and lies in the patch. If no ant then we say empty class.
3. Train a VGG-16 over the patches. Resample the dataset so that the ratio of all samples from all classes is nearly equal. Take care of the samples from empty class specially
