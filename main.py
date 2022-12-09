from model.ddpm import UNet

model1 = UNet()
p1 = sum([p.numel() for p in model1.parameters()])


model2 = UNet(hid_chahhels=128)
p2 = sum([p.numel() for p in model2.parameters()])

print('My model has {} parametets'.format(p1))
print('Their model has {} parameters'.format(p2))
print('It is {} times more....'.format(p2 / p1))