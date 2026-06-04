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
├── Modelo_G2T.ttt
├── lidar.py
├── lidar_kalman.py
└── README.md
```

---

## Archivos principales

| Archivo | Descripción |
|----------|-------------|
| `Modelo_G2T.ttt` | Escena de CoppeliaSim utilizada para modelar y visualizar el sistema G2T (tractor–doble remolque). Permite obtener los datos de referencia (ground truth) empleados en la evaluación de los algoritmos de estimación. |
| `lidar.py` | Procesamiento de datos LiDAR y estimación geométrica de los ángulos φ₁ y φ₂ mediante detección de líneas y análisis de mediciones. |
| `lidar_kalman.py` | Implementación de un Filtro de Kalman Extendido (EKF) para fusionar las mediciones LiDAR con el modelo cinemático del sistema G2T. Incluye comparación entre estimaciones y datos de referencia. |
| `dataset_ROS/` | Rosbag de ROS 2 que contiene las mediciones LiDAR y las variables de referencia utilizadas durante la evaluación. |

---

## Flujo de trabajo

1. Simulación del sistema G2T en CoppeliaSim mediante `Modelo_G2T.ttt`.
2. Generación y almacenamiento de datos en un rosbag de ROS 2.
3. Procesamiento de mediciones LiDAR mediante `lidar.py`.
4. Fusión de sensores utilizando un Filtro de Kalman Extendido mediante `lidar_kalman.py`.
5. Comparación de resultados con los datos de referencia (ground truth).

---

## Requisitos

- Python 3.10 o superior
- ROS 2 bag grabado previamente

---

## Instalación

```bash
pip install numpy matplotlib scikit-learn rosbags --break-system-packages
```

Alternativamente:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scikit-learn rosbags
```

---

## Ejecución

### Estimación geométrica (Parte 1)

```bash
python3 lidar.py --bag ./dataset_ROS
```

### Estimación con EKF (Parte 2)

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

## Simulación en CoppeliaSim

El archivo `Modelo_G2T.ttt` contiene el entorno de simulación utilizado para generar los datos experimentales. La escena modela un sistema tractor–doble remolque (G2T) equipado con sensores LiDAR y permite obtener las variables de referencia necesarias para evaluar el desempeño de los algoritmos de estimación implementados.

---

## Asignatura

**IPD482 — Robótica Móvil Probabilística**  
Universidad Técnica Federico Santa María
