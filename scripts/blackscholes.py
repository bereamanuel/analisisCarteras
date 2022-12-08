from math import log,exp,sqrt
from scipy.stats import norm

class BlackScholes():
    
    def __init__(self,s0,K,T,r,sigma,d0=0,T0=0):
        self.s0 = s0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.d0 = d0
        self.T0 = T0
        self.d1 = None
        self.d2 = None
        self.valorPut = self.valorPut()
        self.valorCall = self.valorCall()

    def __str__(self):
        return f"El valor de la Call: {self.valorCall}\nEl valor de la Put : {self.valorPut}"
    
    def valorCall(self):
        """
        La siguiente función nos devuelve el valor de una Call Europea para fecha de vencimiento T:
        Input:
            - s0(Number)            : Valor inicial de la Call.
            - K(Number)             : Precio del ejercicio.
            - r(Number)             : Tipo de interés libre de riesgo.
            - sigma (Number)        : riesgo o volatilidad del activo.
            - T (Number)            : Fecha de vencimiento.
            - T0 = 0 (Number)       : Fecha inicial para el velor S0.
        Output:
            - C(Number)             : Valor de la call.
        """

        self.d1 = (log(self.s0/self.K) + (self.r + (self.sigma**2)/(2)) * (self.T-self.T0)) / (sqrt(self.T-self.T0)*self.sigma)
        self.d2 = self.d1 - self.sigma*sqrt(self.T-self.T0)

        Nd1 = norm.cdf(self.d1)
        Nd2 = norm.cdf(self.d2)

        C = self.s0*Nd1 - self.K*exp(-self.r*(self.T-self.T0))*Nd2
        return C


    def valorPut(self):
        """
        La siguiente función nos devuelve el valor de una Put Europea para fecha de vencimiento T:
        Input:
            - s0(Number)            : Valor inicial de la Put.
            - K(Number)             : Precio del ejercicio.
            - r(Number)             : Tipo de interés libre de riesgo.
            - sigma (Number)        : riesgo o volatilidad del activo.
            - T (Number)            : Fecha de vencimiento.
            - d0 = 0 (Number)       : Valor de los dividendos
            - T0 = 0 (Number)       : Fecha inicial para el velor S0.
        Output:
            - P(Number)             : Valor de la Put.
        """
        from math import log,exp,sqrt
        from scipy.stats import norm

        self.d1 = (log(self.s0/self.K) + (self.r + (self.sigma**2)/(2)) * (self.T-self.T0)) / (sqrt(self.T-self.T0)*self.sigma)
        self.d2 = self.d1 - self.sigma*sqrt(self.T-self.T0)

        Nd1 = norm.cdf(-self.d1)
        Nd2 = norm.cdf(-self.d2)

        P = - self.s0*Nd1 + self.K*exp(-self.r*(self.T-self.T0))*Nd2 + self.d0
        return P
