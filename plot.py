import os  
import matplotlib.pyplot as plt  
  
# File path  
file_path = "/home/v-wangbohan/mycontainer/ImageNet_Experiment/model/imagenet_lion_sd_1_n_90_bs_128_lr_0.0003_warmup_10000_wd_1.0_hp_[0.9, 0.99]_train.acc.txt"  
  
# Read training accuracy from the text file  
with open(file_path, "r") as file:  
    data = file.readlines()  
  
# Convert strings to floating-point numbers  
training_acc = [float(line.strip()) for line in data]  
  
# Plot the training accuracy  
plt.plot(training_acc)  
plt.xlabel("Epochs")  
plt.ylabel("Training Accuracy")  
plt.title("Training Accuracy vs. Epochs")  
  
# Extract hyperparameter information from the file name  
file_name = os.path.basename(file_path)  
hyperparams = file_name.split("_train.acc.txt")[0]  
  
# Save the figure with hyperparameter information in the name  
plt.savefig(f"./Plot/{hyperparams}_plot.png")  
plt.show()  
