import sympy as sp

sigma = sp.Symbol('\sigma')
epsilon = sp.Symbol('\epsilon')
E, E_0 = sp.symbols('E E_0')
q = sp.Symbol('q')
fwhm = sp.Symbol('\Gamma')

epsilon = (E - E_0)/fwhm

sigma = ((q + epsilon)**2)/(1 + epsilon**2)


sigma = sigma.subs({E_0: 0,  fwhm: 2})

display(sigma)
# plot sigma
sp.plot(sigma, (E, -5, 5), xlabel='E', ylabel='sigma', title='sigma vs E')
