# TP_AA1_Aguirre_Gambande_Paladini

Repositorio de trabajos prácticos de Aprendizaje Automático 1 —
Tecnicatura en Inteligencia Artificial — FCEIA (UNR).

## Trabajos prácticos

- [TP1 — Regresión](tp1-regresion/README.md): predicción de precios de casas.

## Instalación

Entorno compartido para todos los TPs del repo:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Registrar el entorno como kernel de Jupyter (una sola vez):

```bash
python -m ipykernel install --user --name=tp-aa1
```

Luego, al abrir cualquiera de las notebooks, seleccionar el kernel `tp-aa1`.
