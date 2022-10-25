from SPOG import SPOG, SPOGParams, SPOGPoly, SPOGCipher
p = SPOGParams()
p.nphi = 32
p.k = 13
p.t = 256
p.gamma = 256 
p.msk = 131071
p.mtil = 256
cipher = SPOG(p)

a = SPOGPoly()
a.data.append("28")
a.data.append("42")
a.data.append("0")
a.data.append("1")
b = SPOGPoly()
b.data.append("0")
b.data.append("0")
b.data.append("0")
b.data.append("1")

ct = cipher.encrypt(a)