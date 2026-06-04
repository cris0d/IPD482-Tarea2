# IPD482 — Guía 2: Sensores y Estimación del Error

**Robótica Móvil Probabilística**  
Universidad Técnica Federico Santa María · 2026

---

## Integrantes

| Nombre | Correo |
|----------|----------|
| Cristóbal Valenzuela | cristobal.valenzuelb@usm.cl |
| Cristián Ayancán | cristian.ayancan@usm.cl |
| Vicente Osorio | vicente.osorioa@usm.cl |

---

## Descripción

Este proyecto implementa el procesamiento y visualización de datos LiDAR obtenidos desde ROS 2. Se incluyen herramientas para:

- Visualización interactiva de nubes de puntos.
- Detección de líneas mediante RANSAC.
- Agrupamiento de puntos utilizando DBSCAN.
- Análisis geométrico de mediciones LiDAR.

---

## Estructura del proyecto

```text
Tarea2/
├── dataset_ROS/
│   └── ...
├── lidar.py
└── README.md
```

---

## Requisitos

- Python 3.10 o superior
- ROS 2 bag grabado previamente
---

## Instalación

Instalar las dependencias necesarias:

```bash
pip install numpy matplotlib scikit-learn rosbags --break-system-packages
```

Alternativamente, se recomienda utilizar un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install numpy matplotlib scikit-learn rosbags
```

---

## Ejecución

Desde la carpeta raíz del proyecto:

```bash
python3 lidar.py --bag ./dataset_ROS
```

---

## Dataset

El rosbag utilizado debe ubicarse dentro de la carpeta:

```text
dataset_ROS/
```

o bien indicar la ruta correspondiente mediante el parámetro:

```bash
python3 lidar.py --bag <ruta_al_rosbag>
```

---

## Asignatura

**IPD482 — Robótica Móvil Probabilística**  
Universidad Técnica Federico Santa María
