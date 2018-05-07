require 'nngraph'

h1 = - nn.Linear(20,10)
h2 = h1
     - nn.Tanh()
     - nn.Linear(10,10)
     - nn.Tanh()
     - nn.Linear(10, 1)
mlp = nn.gModule({h1}, {h2})

torch.save("model2.t7", mlp)