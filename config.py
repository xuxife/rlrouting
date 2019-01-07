### env ###
IsDrop = True  # whether the network drop packets
Lambda = 2   # the poisson parameter (lambda) of one second
TransTime = 1  # transmission delay of one hop
BandwidthLimit = 3    # the maximum capacity of a connection
SlotsInOneSecond = 3  # the number of time slots in one second

### agent ###
InitQ = 0    # inital value of Q table
InitP = 0
Discount = 0.99  # discount factor
DiscountTrace = 0.6  # discount factor of eligibity trace
DropPenalty = 0  # the penalty of dropping a packet

LearnRateQ = 0.1
LearnRateP = 0.1
