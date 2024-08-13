import torchvision.datasets as ds

ds.MNIST(root="./dataset",train=True,download=True)
ds.MNIST(root="./dataset",train=False,download=True)


