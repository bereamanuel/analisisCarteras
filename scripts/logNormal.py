import yfinance as yf  
import pandas as pd 
import numpy as np
import mplfinance as mpf
import datetime as dt
from math import exp, sqrt
import matplotlib.pyplot as plt

class LogNormal():

    def __init__(self, simbolo = "BBVA.MC", p=30):
        self.acciones = yf.Ticker(simbolo).history(period="max")
        self.datos = self.acciones.iloc[len(self.acciones)-p:][["Open","High","Low","Close"]]
        self.datos.index = range(0,len(self.datos))
        self.em = self.estimacionMomentos()
        self.emv = self.estimacionMV()
        self.emnp = self.estimacionMNP()
        self.error = self.tableE()
        self.p25 = None
        self.muP = None
        self.p975 = None
        self.sigmaP  = None

    def media(self,mu,t = None):
        """
        La siguiente función nos devuelve el media del modelo Log-Normal en el instate t:
        Input:
            - datos(List[Number])   : Datos a analizar.
            - mu(Number)            : Media muestral.
            - t (Number)            : Vector de tiempos o el instante t. (Valor opcional, si no se rellena, devuelve en cada instante de t)
        Output:
            - eS(List[Number])      : Vector de medias de la solución en todos los instantes o media de la solución en el instante t.
        """
        from math import exp
        import pandas as pd 

        s0 = self.datos.iloc[0]
        eS = pd.DataFrame(columns=self.datos.iloc[0].index)
        if t is not None:
            eS = eS.append(pd.DataFrame(s0*list(map(exp,mu*t))).transpose(),ignore_index=True)
        else:
            for t in range(len(self.datos)):
                eS = eS.append(pd.DataFrame(s0*list(map(exp,mu*t))).transpose(),ignore_index=True)
        return(eS)

    def desvTipica(self,mu, sigma, t = None):
        """
        La siguiente función nos devuelve la desviación típica del modelo Log-Normal en el instate t:
        Input:
            - datos(List[Number])   : Datos a analizar.
            - mu(Number)            : Media muestral.
            - sigma(Number)         : Desviación típica muestral.
            - t (Number)            : Vector de tiempos o el instante t. (Valor opcional, si no se rellena, devuelve en cada instante de t)
        Output:
            - eS(List[Number])       : Vector desviación típica la solución en todos los instantes o desviación típica de la solución en el instante t.
        """
        from math import exp, sqrt
        import numpy as np 
        import pandas as pd 

        s0 = self.datos.iloc[0]
        eS = pd.DataFrame(columns=self.datos.iloc[0].index)
        if t is not None:
            eS = eS.append(pd.DataFrame(list(map(sqrt,(s0**2)*list(map(exp,2*mu*t))*list(map(lambda x : exp(x) - 1, (sigma**2)*t)))),index = self.datos.columns).transpose(),ignore_index=True)
        else:
            for t in range(len(self.datos)):
                eS = eS.append(pd.DataFrame(list(map(sqrt,(s0**2)*list(map(exp,2*mu*t))*list(map(lambda x : exp(x) - 1, (sigma**2)*t)))),index = self.datos.columns).transpose(),ignore_index=True)
        return(eS)

    def estimacionMomentos(self):
        """
        La siguiente función nos devuelve mu y sigma obtenidos por el método de los momentos,
        Este método se basa en calcular la media y varianza muestral de la diferencia de logaritmos de nuestros datos, es decir,
        la diferencia de S(t) evaluado en i+1 e i tomando logaritmos. Con esta diferencia obtenemos los valores que nos da el método:
        Input:
            - datos(List[Number])   : Datos a analizar.
        Output:
            - Dict[Number]          : Diccionario con los datos de la media y la desviación típica que nos da el método.
        """
        dt = 1
        u = np.log(self.datos).diff()
        uM = u.mean()
        uV = u.std()**2
        
        return( {"mMME" : ((uM + uV/2)/dt), "sMME" : (np.sqrt(uV/dt)) }   )

    def estimacionMV(self):
        """
        La siguiente función nos devuelve mu y sigma obtenidos por el método de máxima verosimilitud,
        Este método se basa en maximizar la función de verosimilitud.
        Input:
            - datos(List[Number])   : Datos a analizar.
        Output:
            - Dict[Number]          : Diccionario con los datos de la media y la desviación típica que nos da el método.
        """
        n = len(self.datos)
    
        dt = 1

        muE = (1/(n*dt))*np.sum((self.datos/self.datos.shift()) -1)
        siE = np.sqrt((1/(n*dt))*np.sum(((self.datos/self.datos.shift()) -1- muE*dt)**2 ))
        
        return( {"muE" : muE  ,"siE" : siE})

    def estimacionMNP(self):
        """
        La siguiente función nos devuelve mu y sigma obtenidos por el método de momentos no paramétrico.
        Input:
            - datos(List[Number])   : Datos a analizar.
        Output:
            - Dict[Number]          : Diccionario con los datos de la media y la desviación típica que nos da el método.
        """
        n = len(self.datos)
        
        
        dt = 1

        muE = (1/(dt))*(np.sum(self.datos.diff())/np.sum(self.datos[0:(n-1)]))
        siE = np.sqrt((1/(dt))*(np.sum((self.datos.diff())**2)/np.sum((self.datos[0:(n-1)])**2)))
        
        return( {"muMnp" : muE ,"siMnp" : siE})

    def errores(self, modelo):
        """
        La siguiente función nos devuelve los erroes cuadrático medio y absoluto porcentual del modelo ajustado.
        Input:
            - datos(List[Number])   : Datos a analizar.
            - modelo(List[Number])  : Datos que nos devuelve nuestro modelo.
        Output:
            - Dict                  : Diccionario que nos devuelve los valores de error.
        """
        d = np.log(self.datos.iloc[0])
        m = np.log(modelo)

        n = len(self.datos)

        ecm = np.sqrt((1/n)*(np.sum( (d-m)**2 )) )

        mape = 100/n * np.sum(np.abs(d-m)/np.abs(d))

        return({"ECM": ecm.to_dict() , "MAPE" : mape.to_dict()})

    def validacion(self,estimacion):
        """
        La siguiente función nos devuleve los errores cometidos al utilizar cierto estimador. Se complementa con la función error.
        Input:
            - estimacion(Dict[Number]) : Diccionario que contiene la media y la sigma estimada con cierta función de estimación.
            - datos(List[Number])   : Datos a analizar.
        Output:
            - Dict                  : Diccionario que nos devuelve los valores de error.
        """
        mu = [estimacion[k] for k in estimacion.keys() if k in ("mMME","muE","muMnp")][0]
        sigma = [estimacion[k] for k in estimacion.keys() if k not in ("mMME","muE","muMnp")][0]
        mm = self.media(mu)

        return(self.errores(mm))

    def predecir(self, est, t):
        """
        La siguiente función nos devuleve la tabla con los valores obtenidos en nuestro modelo. Además contiene la prediccion en las ultimas t filas.
        Input:
            - datos(List[Number])   : Datos a analizar.
            - est(Dict[Number])     : Diccionario que contiene la media y la sigma estimada con cierta función de estimación.
            - t (Number)            : Numero de periodos que queremos estimar.
        Output:
            - tabla(DataFrame)      : Diccionario que nos devuelve los valores de error.
        """
        mu = [est[k] for k in est.keys() if k in ("mMME","muE","muE")][0]
        sigma = [est[k] for k in est.keys() if k not in ("mMME","muE","muE")][0]

        val = self.validacion(est)

        muP = self.media(mu)
        for i in range(0,t):
            muP = muP.append(self.media(mu,len(self.datos)+i),ignore_index=True)

        self.muP = muP

        sigmaP = self.desvTipica(mu, sigma)
        for i in range(0,t):
            sigmaP = sigmaP.append(self.desvTipica(mu,sigma,len(self.datos)+i),ignore_index=True)

        self.sigmaP = sigmaP

        self.p25 = pd.DataFrame(np.array(muP)-1.96*np.array(sigmaP), columns = self.datos.columns)
        self.p975 = pd.DataFrame(np.array(muP) + 1.96*np.array(sigmaP), columns = self.datos.columns)
        print( f"ECM -> {val['ECM']}\n- - - - - - - \nMAPE -> {val['MAPE']}%")


    def graficar(self,est, t):
        """
        La siguiente función nos grafica los resultados del modelo
        Input:
            - datos(List[Number])   : Datos a analizar.
            - est(Dict[Number])     : Diccionario que contiene la media y la sigma estimada con cierta función de estimación.
            - t (Number)            : Numero de periodos que queremos estimar.
        Output:
            - Información del modelo + Gráficos
        """
        self.predecir(est, t)

        for col in self.datos.columns:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,4))
            ax1.plot(self.datos[col], label = "Datos", color = 'green')
            ax1.plot(self.p25[col], label = "IC 95% Inf", color = 'tomato')
            ax1.plot(self.muP[col], label = "Media", color = 'red' ,linestyle= 'dashed')
            ax1.plot(self.p975[col], label = "IC 95% Sup", color = 'coral')
            ax1.axvline(x = len(self.datos)-1 , color = 'black') 
            ax1.legend()

            ax1.set_ylabel("Price")
            ax1.set_xlabel("t")
            
            ax1.set_title("Datos vs Modelo")


            ax2.plot(self.muP[col].cumsum(), label = "Media", color = 'red' ,linestyle= 'dashed')
            ax2.plot(self.datos[col].cumsum(), label = "Datos", color = 'green')
            ax2.axvline(x = len(self.datos)-1 , color = 'black') 

            ax2.set_ylabel("Cumulative price")
            ax2.set_xlabel("t")
            
            ax2.set_title("Datos vs Modelo cumulative")

            fig.suptitle("Datos " + col)

            plt.show()

    def tableE(self):

        valMom = pd.DataFrame(self.validacion(self.em))
        valMV = pd.DataFrame(self.validacion(self.emv))
        valMNP = pd.DataFrame(self.validacion(self.emnp))

        return pd.concat([valMom, valMV,valMNP], keys=['M', 'Mv','MNP'])
