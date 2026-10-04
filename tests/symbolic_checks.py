"""Symbolic checks of the derivations (copied from scripts/transitions/math_checks.py); collected by tests/test_symbolic.py."""
import sympy as sp

xi, v1, v2, T, c0, c1, c2, c3 = sp.symbols("xi v1 v2 T c0 c1 c2 c3", real=True)
vf, vq, L, w, S, x, n, fd, C, kj, om, beta, y = sp.symbols("v_f v_q L w S x n f_d C k_j omega beta y", positive=True)
ok = {}

# KS1: boundary conditions, beta identity, moments, displacement, 9/140
h = 3 * xi**2 - 2 * xi**3
ok["KS1 boundary conditions h(0)=0,h(1)=1,h'(0)=h'(1)=0"] = [h.subs(xi, 0), h.subs(xi, 1), sp.diff(h, xi).subs(xi, 0), sp.diff(h, xi).subs(xi, 1)] == [0, 1, 0, 0]
ok["h = I_xi(2,2) (integral of Beta(2,2) density)"] = sp.simplify(sp.integrate(6 * xi * (1 - xi), (xi, 0, xi)) - h) == 0
ok["moments 1/2, 13/35, 43/140"] = [sp.integrate(h**k, (xi, 0, 1)) for k in (1, 2, 3)] == [sp.Rational(1, 2), sp.Rational(13, 35), sp.Rational(43, 140)]
dv = v2 - v1
v = v1 + dv * h
ok["displacement (v1+v2)T/2"] = sp.simplify(T * sp.integrate(v, (xi, 0, 1)) - (v1 + v2) * T / 2) == 0
r = lambda s: c0 + c1 * s + c2 * s**2 + c3 * s**3
de = T * sp.integrate(r(v), (xi, 0, 1)) - T / 2 * (r(v1) + r(v2))
rpp = lambda s: sp.diff(r(sp.Symbol("s")), sp.Symbol("s"), 2).subs(sp.Symbol("s"), s)
ok["delta e = -(9/140) T dv^2 r''(vbar)"] = sp.simplify(de + sp.Rational(9, 140) * T * dv**2 * rpp((v1 + v2) / 2)) == 0
ok["intermediate: dv^2[r''(v1)+3 c3 dv] = dv^2 r''(vbar)"] = sp.simplify(rpp(v1) + 3 * c3 * dv - rpp((v1 + v2) / 2)) == 0
a = sp.Symbol("a", positive=True)
ok["cube law -27/280 |dv|^3 r''/a with T = 3|dv|/(2a)"] = sp.simplify(sp.Rational(9, 140) * (3 * S / (2 * a)) * S**2 - sp.Rational(27, 280) * S**3 / a) == 0

# closed queue: lambda*w = mu*w + (mu/2) d(w^2)/dt with lambda = mu(1+w')
t, mu = sp.symbols("t mu", positive=True)
W = sp.Function("w")(t)
ok["lambda w = mu w + (mu/2) d(w^2)/dt"] = sp.simplify(mu * (1 + sp.diff(W, t)) * W - (mu * W + mu / 2 * sp.diff(W**2, t))) == 0

# cubic Gamma and psi''
G = (vf * r(vq) - vq * r(vf)) / (vf - vq)
ok["Gamma(cubic) = c0 - c2 vf vq - c3 vf vq (vf+vq)"] = sp.simplify(G - (c0 - c2 * vf * vq - c3 * vf * vq * (vf + vq))) == 0
p = sp.Symbol("p", positive=True)
psi = p * r(1 / p)
ok["psi'' = 2(c2 p + 3 c3)/p^4"] = sp.simplify(sp.diff(psi, p, 2) - 2 * (c2 * p + 3 * c3) / p**4) == 0
ok["chord slope in pace equals Gamma"] = sp.simplify((psi.subs(p, 1 / vq) - psi.subs(p, 1 / vf)) / (1 / vq - 1 / vf) - G) == 0

# transition window (Eq. tradmiss): T_queue >= S/2 and T_free >= S/2
Tq = vf * w / (vf - vq); Tfr = L / vf - vq * w / (vf - vq)
lo = sp.solve(sp.Eq(Tq, S / 2), w)[0]; hi = sp.solve(sp.Eq(Tfr, S / 2), w)[0]
ok["lower bound (vf-vq)S/(2vf)"] = sp.simplify(lo - (vf - vq) * S / (2 * vf)) == 0
ok["upper bound L(1/vq-1/vf)-(vf-vq)S/(2vq)"] = sp.simplify(hi - (L * (1 / vq - 1 / vf) - (vf - vq) * S / (2 * vq))) == 0

# queue-speed elasticity on the power branch: x vq'/vq = (1-n) kj/kq, kq = mu/vq, mu = C x^(1-n)/fd
vqx = C / (kj * fd * x**(n - 1) - C / om)
muq = C * x**(1 - n) / fd
kq = muq / vqx
ok["x vq'(x)/vq = (1-n) kj/kq"] = sp.simplify(x * sp.diff(vqx, x) / vqx - (1 - n) * kj / kq) == 0
ok["kq = kj - mu/omega (triangular FD)"] = sp.simplify(kq - (kj - muq / om)) == 0

# Pigou PoA
ySO = (beta + 1)**(-1 / beta)
Jso = ySO**(beta + 1) + 1 - ySO
ok["SO condition (beta+1) y^beta = 1"] = sp.simplify(sp.diff(y**(beta + 1) + 1 - y, y).subs(y, ySO)) == 0
ok["PoA = [1 - beta(beta+1)^(-(beta+1)/beta)]^-1"] = sp.simplify(1 / Jso - 1 / (1 - beta * (beta + 1)**(-(beta + 1) / beta))) == 0
ok["PoA(1) = 4/3"] = sp.nsimplify((1 / Jso).subs(beta, 1)) == sp.Rational(4, 3)

# transitions: demand-marginal emission, E(D) = D e(D/C), e(x) = e0 + Tf alpha x^beta Gamma(x)
Dd, e0s, Tfs, al = sp.symbols("D e_0 T_f alpha", positive=True)
Gf = sp.Function("Gamma")
xs = Dd / C
e_of_x = lambda z: e0s + Tfs * al * z**beta * Gf(z)
E = Dd * e_of_x(xs)
lhs = sp.diff(E, Dd)
xe_prime = Tfs * al * xs**beta * (beta * Gf(xs) + xs * sp.Subs(sp.diff(Gf(x), x), x, xs).doit())
ok["dE/dD = e(x) + x e'(x) with x e' = Tf alpha x^beta [beta Gamma + x Gamma']"] = sp.simplify(lhs - (e_of_x(xs) + xe_prime)) == 0

