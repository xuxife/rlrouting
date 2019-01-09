### env ###
IsDrop = True  # whether the network drop packets
Lambda = 2   # the poisson parameter (lambda) of one second
TransTime = 1  # transmission delay of one hop
BandwidthLimit = 3    # the maximum capacity of a connection
SlotTime = 1/3  # the time of one slot (an unit of action)

### agent ###
InitQ = 0    # inital value of Q table
InitP = 0
Discount = 0.99  # discount factor
DiscountTrace = 0.6  # discount factor of eligibity trace
DropPenalty = -10  # the penalty of dropping a packet

LearnRateQ = 0.1
LearnRateP = 0.1
