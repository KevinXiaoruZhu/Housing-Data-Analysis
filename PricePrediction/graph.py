import matplotlib.pyplot as plt

font = {'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 20,
        }

x = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
y = [3.49, 2.43, 2.06, 1.75, 1.42, 1.25, 1.16]

plt.figure(1, figsize=(16, 9))
plt.plot(x, y, linewidth='6.0')
plt.xscale('log')
plt.xlabel('iteration', fontdict=font)
plt.ylabel('RMSLE', fontdict=font)
plt.tick_params(labelsize=20)
plt.show()
# fg.savefig('test2png.png', dpi=100)

plt.figure(2, figsize=(16, 9))
plt.plot(x, y, linewidth='6.0')
plt.xscale('log')
plt.xlabel('iteration', fontdict=font)
plt.ylabel('RMSLE', fontdict=font)
plt.tick_params(labelsize=20)
plt.show()