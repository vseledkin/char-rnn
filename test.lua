require "nn"
ms = nn.MaskedSelect()
mask = torch.FloatTensor({{2.99, 1}, {0, 0}}):byte()
input = torch.DoubleTensor({{10, 20}, {30, 40}})
print(input)
print(mask)
out = ms:forward({input, mask})
print(out)
