# view_mnist.py
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 下載 / 載入 MNIST
tf = transforms.ToTensor()  # 轉 tensor 並自動正規化到 [0,1]
mnist_train = datasets.MNIST("data", train=True, download=True, transform=tf)

print(f"Dataset size: {len(mnist_train)} samples")

# 取前 9 筆出來看看
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
for i in range(9):
    img, label = mnist_train[i]
    # img shape: [1,28,28] → squeeze 去掉 channel
    axes[i//3, i%3].imshow(img.squeeze(), cmap="gray")
    axes[i//3, i%3].set_title(f"Label: {label}")
    axes[i//3, i%3].axis("off")

plt.tight_layout()
plt.show()
