import pandas as pd
import json as json
from io import StringIO


def loadDataFrame(path: str) -> pd.DataFrame:
    """
       Carica il dataframe in formato R Data Frame e restituisce un oggetto pandas.DataFrame
    """

    # Apro il file in sola lettura e mi assicuro di chiuderlo una volta finito
    with open(path, 'r') as raw:

        # Leggo e splitto header e body
        string = raw.read().split('@data')

        # Leggo solamente le etichette delle colonne
        head = string[0].split('\n')[2:]
        head = [h.split(' ')[1] for h in head if h]

        # Leggo il body
        body = string[1][1:]

        # Creo un oggetto StringIO per poterlo leggere con pandas
        data = '{}\n{}'.format(','.join(head), body)
        data = StringIO(data)

        # Creo e restituisco il dataframe
        return pd.read_csv(data, sep=',')


def saveBestParameterModel(model_name,param):
    #aggiungo il nome del modello al dizionario
    param.update({'model_name':model_name})

    filename = '../data/best_parameter_models.json'
    listObj = []
 
    # leggo il file JSON 
    with open(filename) as fp:
        listObj = json.load(fp)   
    
    #controllo se il modello è già presente nella lista con i parametri aggiornati
    is_already_present = False
    for i in range(len(listObj)):
        #se è presente aggiorno i parametri
        if listObj[i]['model_name'] == model_name:
            is_already_present = True
            for key in param.keys():
                listObj[i][key] = param[key]
    
    #se non è presente lo aggiungo ex-novo
    if not is_already_present:
        listObj.append(param)
    
    
    # scrivo il file JSON
    with open(filename, 'w') as json_file:
        json.dump(listObj, json_file, 
                            indent=4,  
                            separators=(',',': '))

def loadBestParameterModel(model_name):
    filename = './data/best_parameter_models.json'
    listObj = []
 
    # leggo il file JSON
    with open(filename) as fp:
        listObj = json.load(fp)   
    
    #restituisco il dizionario con i parametri del modello
    for i in range(len(listObj)):
        if listObj[i]['model_name'] == model_name:
            del listObj[i]['model_name']
            return listObj[i]
    
    return None
