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
- Fusión de sensores mediante Filtro de Kalman Extendido (EKF).

---

## Estructura del proyecto

```text
Tarea2/
├── dataset_ROS/
│   └── ...
├── lidar.py
├── lidar_kalman.py
└── README.md
```

---

## Scripts

| Script | Descripción |
|--------|-------------|
| `lidar.py` | Estimación geométrica de ángulos φ₁ y φ₂ con visualización interactiva y evaluación de error. |
| `lidar_kalman.py` | Extensión con Filtro de Kalman Extendido (EKF) para fusión de sensores. Incluye modelo cinemático del sistema G2T y comparación EKF vs ground truth. |

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

Estimación geométrica (Parte 1):

```bash
python3 lidar.py --bag ./dataset_ROS
```

Estimación con EKF (Parte 2):

```bash
python3 lidar_kalman.py --bag ./dataset_ROS
```

---

## Dataset

El rosbag debe ubicarse dentro de la carpeta `dataset_ROS/`, o bien indicar la ruta mediante:

```bash
python3 lidar.py --bag <ruta_al_rosbag>
```

---

## Asignatura

**IPD482 — Robótica Móvil Probabilística**  
Universidad Técnica Federico Santa María