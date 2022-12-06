# script for jupyter notebook
import yfinance as yf  
import pandas as pd 
import numpy as np
import mplfinance as mpf
import datetime as dt
from math import exp

import matplotlib.pyplot as plt

def media(datos,mu,t = None):
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

        s0 = datos.iloc[0]
        eS = pd.DataFrame(columns=datos.iloc[0].index)
        if t:
            eS = eS.append(pd.DataFrame(s0*list(map(exp,mu*t))).transpose(),ignore_index=True)
        else:
            for t in range(len(datos)):
                eS = eS.append(pd.DataFrame(s0*list(map(exp,mu*t))).transpose(),ignore_index=True)
        return(eS)
def desvTipica(datos ,mu, sigma, t = None):
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

    s0 = datos.iloc[0]
    eS = pd.DataFrame(columns=datos.iloc[0].index)
    if t:
        eS = eS.append(pd.DataFrame(list(map(sqrt,(s0**2)*list(map(exp,2*mu*t))*list(map(lambda x : exp(x) - 1, (sigma**2)*t)))),index = datos.columns).transpose(),ignore_index=True)
    else:
        for t in range(len(datos)):
            eS = eS.append(pd.DataFrame(list(map(sqrt,(s0**2)*list(map(exp,2*mu*t))*list(map(lambda x : exp(x) - 1, (sigma**2)*t)))),index = datos.columns).transpose(),ignore_index=True)
    return(eS)
def estimacionMomentos(datos):
    """
    La siguiente función nos devuelve mu y sigma obtenidos por el método de los momentos,
    Este método se basa en calcular la media y varianza muestral de la diferencia de logaritmos de nuestros datos, es decir,
    la diferencia de S(t) evaluado en i+1 e i tomando logaritmos. Con esta diferencia obtenemos los valores que nos da el método:
    Input:
        - datos(List[Number])   : Datos a analizar.
    Output:
        - Dict[Number]          : Diccionario con los datos de la media y la desviación típica que nos da el método.
    """
    import numpy as np
    import pandas as pd
    import datetime as dt

    try:
        dt = (datos.index[1] - datos.index[0]).days
        u = np.log(datos).diff()
        uM = u.mean()
        uV = u.std()**2
        
        return( {"mMME" : (uM + uV/2)/dt , "sMME" : np.sqrt(uV/dt) }   )
    except:
        dt = 1
        u = np.log(datos).diff()
        uM = u.mean()
        uV = u.std()**2
        
        return( {"mMME" : (uM + uV/2)/dt , "sMME" : np.sqrt(uV/dt) }   )
def estimacionMV(datos):
    """
    La siguiente función nos devuelve mu y sigma obtenidos por el método de máxima verosimilitud,
    Este método se basa en maximizar la función de verosimilitud.
    Input:
        - datos(List[Number])   : Datos a analizar.
    Output:
        - Dict[Number]          : Diccionario con los datos de la media y la desviación típica que nos da el método.
    """
    import numpy as np
    import pandas as pd
    import datetime as dt
    n = len(datos)
    try:
        dt = (datos.index[1] - datos.index[0]).days

        muE = (1/(n*dt))*np.sum((datos/datos.shift()) -1)
        siE = np.sqrt((1/(n*dt))*np.sum(((datos/datos.shift()) -1- muE*dt)**2 ))
        
        return( {"muE" : muE  ,"siE" : siE})
    except:
        dt = 1

        muE = (1/(n*dt))*np.sum((datos/datos.shift()) -1)
        siE = np.sqrt((1/(n*dt))*np.sum(((datos/datos.shift()) -1- muE*dt)**2 ))
        
        return( {"muE" : muE  ,"siE" : siE})
def estimacionMNP(datos):
    """
    La siguiente función nos devuelve mu y sigma obtenidos por el método de momentos no paramétrico.
    Input:
        - datos(List[Number])   : Datos a analizar.
    Output:
        - Dict[Number]          : Diccionario con los datos de la media y la desviación típica que nos da el método.
    """
    import numpy as np
    import pandas as pd
    import datetime as dt
    n = len(datos)
    
    try:
        dt = (datos.index[1] - datos.index[0]).days
        muE = (1/(dt))*(np.sum(datos.diff())/np.sum(datos[0:(n-1)]))
        siE = np.sqrt((1/(dt))*(np.sum((datos.diff())**2)/np.sum((datos[0:(n-1)])**2)))
        
        return( {"muMnp" : muE  ,"siMnp" : siE})
    except:
        dt = 1

        muE = (1/(dt))*(np.sum(datos.diff())/np.sum(datos[0:(n-1)]))
        siE = np.sqrt((1/(dt))*(np.sum((datos.diff())**2)/np.sum((datos[0:(n-1)])**2)))
        
        return( {"muMnp" : muE  ,"siMnp" : siE})
def errores(datos, modelo):
    """
    La siguiente función nos devuelve los erroes cuadrático medio y absoluto porcentual del modelo ajustado.
    Input:
        - datos(List[Number])   : Datos a analizar.
        - modelo(List[Number])  : Datos que nos devuelve nuestro modelo.
    Output:
        - Dict                  : Diccionario que nos devuelve los valores de error.
    """
    import numpy as np
    import pandas as pd

    d = np.log(datos.iloc[0])
    m = np.log(modelo)

    n = len(datos)

    ecm = np.sqrt((1/n)*(np.sum( (d-m)**2 )) )

    mape = 100/n * np.sum(np.abs(d-m)/np.abs(d))

    return({"ECM": ecm , "MAPE" : mape})
def validacion(estimacion, datos):
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

    mm = media(datos,mu)

    return(errores(datos, mm))
def predecir(datos, est, t):
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

    val = validacion(est, datos)

    muP = media(datos,mu)
    for i in range(0,t):
        muP = muP.append(media(datos,mu,len(datos)+i),ignore_index=True)

    sigmaP = desvTipica(datos,mu, sigma)
    for i in range(0,t):
        sigmaP = sigmaP.append(desvTipica(datos,mu,sigma,len(datos)+i),ignore_index=True)


    p25 = pd.DataFrame(np.array(muP)-1.96*np.array(sigmaP), columns = datos.columns)
    p975 = pd.DataFrame(np.array(muP) + 1.96*np.array(sigmaP), columns = datos.columns)
        


    print( f"ECM -> {val['ECM']}\n- - - - - - - \nMAPE -> {val['MAPE']}%\n- - - - -  - \nPrediccion a {t} días.")

    return(p25,muP,p975,sigmaP)
def graficar(datos, est, t):
    """
    La siguiente función nos grafica los resultados del modelo
    Input:
        - datos(List[Number])   : Datos a analizar.
        - est(Dict[Number])     : Diccionario que contiene la media y la sigma estimada con cierta función de estimación.
        - t (Number)            : Numero de periodos que queremos estimar.
    Output:
        - Información del modelo + Gráficos
    """
    p25,muP,p975,sigmaP = predecir(datos, est, t)

    for col in datos.columns:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,4))
        ax1.plot(datos[col], label = "Datos", color = 'green')
        ax1.plot(p25[col], label = "IC 95% Inf", color = 'tomato')
        ax1.plot(muP[col], label = "Media", color = 'red' ,linestyle= 'dashed')
        ax1.plot(p975[col], label = "IC 95% Sup", color = 'coral')
        ax1.axvline(x = len(datos)-1 , color = 'black') 
        ax1.legend()

        ax1.set_ylabel("Price")
        ax1.set_xlabel("t")
        
        ax1.set_title("Datos vs Modelo")


        ax2.plot(muP[col].cumsum(), label = "Media", color = 'red' ,linestyle= 'dashed')
        ax2.plot(datos[col].cumsum(), label = "Datos", color = 'green')
        ax2.axvline(x = len(datos)-1 , color = 'black') 

        ax2.set_ylabel("Cumulative price")
        ax2.set_xlabel("t")
        
        ax2.set_title("Datos vs Modelo cumulative")

        fig.suptitle("Datos " + col)

        plt.show()
