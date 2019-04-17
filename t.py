class A:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __repr__(self):
        return f"<A a={self.a} b={self.b}>"
    def __lt__(self, other):
        return self.a < other.a

l = [A(i, 'a') for i in range(10, 1, -1)]
b = []
for i in l:
    heappush(b, i)
