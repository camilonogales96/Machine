import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def buscar_archivo(nombre_archivo):
    """Busca un archivo recursivamente hacia arriba desde BASE_DIR."""
    curr = BASE_DIR
    while curr:
        cand = os.path.join(curr, nombre_archivo)
        if os.path.exists(cand):
            return cand
        cand_sub = os.path.join(curr, 'redwine', nombre_archivo)
        if os.path.exists(cand_sub):
            return cand_sub
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return os.path.join(BASE_DIR, nombre_archivo)

def limpiar_dataset_hr():
    """Limpia HRDataset_v14.csv y exporta dataset_hr_limpio.csv."""
    ruta_origen = buscar_archivo('HRDataset_v14.csv')
    if not os.path.exists(ruta_origen):
        print(f"[ERROR] No se encontro {ruta_origen}")
        return None
    
    df = pd.read_csv(ruta_origen)
    df = df.drop_duplicates()
    df = df.dropna(subset=['Termd'])
    
    # Seleccionar columnas numericas y excluir IDs no predictivas si existen
    variables_numericas = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    df_num = df[variables_numericas].fillna(0)
    
    objetivo = df_num['Termd']
    columnas_descartar = ['Termd']
    if 'EmpID' in df_num.columns:
        columnas_descartar.append('EmpID')
        
    variables_entrada = df_num.drop(columns=columnas_descartar)
    
    normalizador = StandardScaler()
    valores_escalados = normalizador.fit_transform(variables_entrada)
    datos_normalizados = pd.DataFrame(valores_escalados, columns=variables_entrada.columns)
    datos_normalizados['Termd'] = objetivo.values
    
    ruta_destino = os.path.join(BASE_DIR, 'dataset_hr_limpio.csv')
    datos_normalizados.to_csv(ruta_destino, index=False)
    print(f"[LIMPIEZA HR] Dataset limpio exportado exitosamente en: {ruta_destino}")
    return datos_normalizados

def limpiar_dataset_vinos():
    """Limpia winequality-red.csv y exporta dataset_vino_limpio.csv."""
    ruta_origen = buscar_archivo('winequality-red.csv')
    if not os.path.exists(ruta_origen):
        print(f"[ERROR] No se encontro {ruta_origen}")
        return None
        
    df = pd.read_csv(ruta_origen)
    df = df.drop_duplicates().dropna()
    
    objetivo = df['quality']
    variables_entrada = df.drop('quality', axis=1)
    
    normalizador = StandardScaler()
    valores_escalados = normalizador.fit_transform(variables_entrada)
    datos_normalizados = pd.DataFrame(valores_escalados, columns=variables_entrada.columns)
    datos_normalizados['quality'] = objetivo.values
    
    ruta_destino = os.path.join(BASE_DIR, 'dataset_vino_limpio.csv')
    datos_normalizados.to_csv(ruta_destino, index=False)
    print(f"[LIMPIEZA VINOS] Dataset limpio exportado exitosamente en: {ruta_destino}")
    return datos_normalizados

def ejecutar_limpieza_completa():
    df_hr = limpiar_dataset_hr()
    df_vinos = limpiar_dataset_vinos()
    return df_hr, df_vinos

if __name__ == '__main__':
    ejecutar_limpieza_completa()
